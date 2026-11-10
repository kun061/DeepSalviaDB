import json
import os
import logging
import mysql.connector
from typing import Dict, Any, List
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import decimal

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXPRESSION_TABLE_RE = re.compile(
    r"\b(?:salvia_[a-z0-9_]+_(?:tissue_expression|abstress)_(?:fpkm|count)"
    r"|multi_species_rpkm)\b",
    re.IGNORECASE,
)

QUALIFIED_COLUMN_EQUALITY_RE = re.compile(
    r"(?<!BINARY\s)"
    r"(?P<left>`?[A-Za-z_][A-Za-z0-9_$]*`?\.`?[A-Za-z_][A-Za-z0-9_$]*`?)"
    r"\s*=\s*"
    r"(?P<right>`?[A-Za-z_][A-Za-z0-9_$]*`?\.`?[A-Za-z_][A-Za-z0-9_$]*`?)",
    re.IGNORECASE,
)


def _repair_collation_equalities(query: str) -> str:
    """Use byte-exact comparison for qualified column-to-column equalities."""
    return QUALIFIED_COLUMN_EQUALITY_RE.sub(
        lambda match: f"BINARY {match.group('left')} = BINARY {match.group('right')}",
        query,
    )


def _normalize_generated_sql(query: str, question: str) -> str:
    """Repair narrow, schema-invariant Text-to-SQL mistakes."""
    normalized = re.sub(
        r"\bmean_root_fresh_weight\b",
        "mean_fresh_weight_g",
        query,
        flags=re.IGNORECASE,
    )
    query = normalized

    normalized = re.sub(r"\bprimary_root_length\b", "root_length", query, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\bmean_primary_root_length_cm\b",
        "mean_root_length_cm",
        normalized,
        flags=re.IGNORECASE,
    )

    question_lower = question.casefold()
    # Preserve sample-level phenotype semantics instead of broadening to a stage aggregate.
    phenotype_sample = re.search(r"\b(DAG(?:15|30|45|60|75|90|105|120|150|180|210|240|270|300|330|360|390)-\d+)\b", question, re.IGNORECASE)
    phenotype_fields = (
        (r"\btotal root volume\b", "total_root_volume", "cm^3"),
        (r"\btotal root surface area\b", "total_root_surface_area", "cm^2"),
        (r"\b(?:number of )?root forks?\b", "number_of_root_forks", "count"),
        (r"\baverage root diameter\b", "average_root_diameter", "mm"),
        (r"\bfresh weight\b", "fresh_weight", "g"),
        (r"\b(?:primary )?root length\b", "root_length", "cm"),
    )
    if phenotype_sample and "root_phenotype_data" in normalized.casefold():
        for label, field, unit in phenotype_fields:
            if re.search(label, question, re.IGNORECASE):
                sample_id = phenotype_sample.group(1).upper()
                normalized = (
                    f"SELECT sample_id, dag_stage, replicate, {field}, '{unit}' AS unit "
                    f"FROM root_phenotype_data WHERE sample_id = '{sample_id}'"
                )
                break

    # Cultivated-metabolomics abundance spans the abundance, compound, and species tables.
    cultivated = re.search(
        r"\b(?:for|of)\s+(.+?)\s+(?:in|for|observed\s+in)\s+(?:the\s+)?"
        r"(Salvia(?:[ _][A-Za-z][A-Za-z0-9_-]*)+?)(?=\s+(?:sample|accession)\b|[?.!,]|$)",
        question, re.IGNORECASE,
    )
    if cultivated and ("abundance" in question_lower or "measured level" in question_lower) and not re.search(r"\bDAG\s*\d+\b", question, re.IGNORECASE):
        compound_name = cultivated.group(1).strip().replace("'", "''")
        compound_name = re.sub(r"^(?:compound|metabolite)\s+", "", compound_name, flags=re.IGNORECASE)
        species_name = cultivated.group(2).strip().replace("'", "''")
        normalized = (
            "SELECT c.compound_id, c.name, s.sample_id, s.scientific_name, a.abundance "
            "FROM compound_abundance a "
            "JOIN salvia_compounds c ON BINARY a.compound_id = BINARY c.compound_id "
            "JOIN salvia_species s ON BINARY a.sample_id = BINARY s.sample_id "
            f"WHERE c.name = '{compound_name}' AND s.scientific_name LIKE '{species_name}%'"
        )


    # Treatment samples live in abstress matrices even when their names also
    # contain a tissue keyword such as leaf or root.
    if re.search(r"(?:^|[^a-z0-9])(?:stress|light|heat|cold|drought|treatment|naa|aba|salt)(?:[^a-z0-9]|$)", question_lower):

        normalized = re.sub(
            r"\b(salvia_[a-z0-9_]+)_tissue_expression_(fpkm|count)\b",
            r"\1_abstress_\2",
            normalized,
            flags=re.IGNORECASE,
        )

    # multi_species_rpkm is a wide matrix: species are columns, not rows.
    if "multi_species_rpkm" in normalized.casefold():

        normalized = re.sub(
            r"\s+(?:AND|OR)\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?species_name\s*"
            r"(?:=|LIKE)\s*'[^']*'",
            "",
            normalized,
            flags=re.IGNORECASE,
        )

    # Direct DAG columns belong to salvia_metabolite_info; replicate-suffixed
    # DAG columns belong only to salvia_hormone_profilling.
    if "salvia_metabolite_info" in normalized.casefold():
        requested_stages = re.findall(
            r"\bDAG\s*(30|60|90|240|270|390)\b", question, re.IGNORECASE
        )
        requested_stages = list(dict.fromkeys(requested_stages))
        if requested_stages and "abundance" in question_lower:
            stage = requested_stages[0]
            select_clause = f"SELECT compound_id, name, dag{stage} AS abundance FROM salvia_metabolite_info"
            if len(requested_stages) >= 2 and re.search(
                r"\b(?:compare|contrast|difference|minus|latter|change)\b", question, re.IGNORECASE
            ):
                later_stage = requested_stages[1]
                select_clause = (
                    f"SELECT compound_id, name, dag{stage} AS value_a, "
                    f"dag{later_stage} AS value_b, "
                    f"(dag{later_stage} - dag{stage}) AS absolute_change "
                    "FROM salvia_metabolite_info"
                )
            normalized = re.sub(
                r"^\s*SELECT\s+.+?\s+FROM\s+salvia_metabolite_info\b",
                select_clause, normalized, count=1, flags=re.IGNORECASE | re.DOTALL,
            )


        normalized = re.sub(
            r"\bcompound_id\s+LIKE\s+('%[^']+%')",
            r"name LIKE \1",
            normalized,
            flags=re.IGNORECASE,
        )

        normalized = re.sub(
            r"\bdag(30|60|90|240|270|390)_[123]\b",
            r"dag\1",
            normalized,
            flags=re.IGNORECASE,
        )

        normalized = re.sub(
            r"\bcompound_id\s*(=|LIKE)\s*'(%?[^']*[a-z][^']*\s+[^']*%?)'",
            lambda match: f"name {match.group(1)} '{match.group(2)}'",
            normalized,
            flags=re.IGNORECASE,
        )

        # Do not invent a short compound ID from a natural-language name.
        invalid_id = re.search(
            r"\bcompound_id\s*=\s*'(?!Com_\d+_(?:pos|neg)\b)[^']+'",
            normalized,
            re.IGNORECASE,
        )
        named_abundance = re.search(
            r"\b(?:report|and|of|for|the)\s+"
            r"([A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,3})\s+abundance\b",
            question,
            re.IGNORECASE,
        )
        if invalid_id and named_abundance:
            metabolite_name = named_abundance.group(1).replace("'", "''")

            normalized = re.sub(
                r"\bcompound_id\s*=\s*'(?!Com_\d+_(?:pos|neg)\b)[^']+'",
                f"name LIKE '%{metabolite_name}%'",
                normalized,
                flags=re.IGNORECASE,
            )

    # A composite identifier containing "&" is one literal database ID.
    composite = re.search(r"\b([A-Za-z0-9_.:-]+&[A-Za-z0-9_.:-]+)\b", question)
    if composite and re.search(r"\b(?:variant|SNP|Indel|impact).*(?:count|category)|\bcounts?\b", question, re.IGNORECASE):
        literal = composite.group(1).replace("'", "''")
        normalized = (
            "SELECT gene_id, chrom, start_pos, end_pos, snp_count, indel_count, "
            "impact_high, impact_moderate, impact_low, missense_variant, "
            "synonymous_variant FROM salvia_db_gene_summary "
            f"WHERE gene_id = '{literal}'"
        )
    elif composite and "salvia_db_gene_summary" in normalized.casefold():
        literal = composite.group(1).replace("'", "''")
        normalized = re.sub(
            r"\b((?:[A-Za-z_][A-Za-z0-9_]*\.)?gene_id)\s+IN\s*\([^)]*\)",
            lambda match: f"{match.group(1)} = '{literal}'",
            normalized,
            flags=re.IGNORECASE,
        )

    # Expression matrices are wide: a sample is a column, not a row value.
    sample_match = re.search(
        r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?sample_id\s*=\s*'([A-Za-z0-9_]+)'",
        normalized,
        re.IGNORECASE,
    )
    table_match = EXPRESSION_TABLE_RE.search(normalized)
    if sample_match and table_match:
        sample_column = sample_match.group(1)
        table_name = table_match.group(0).casefold()
        metric = "rpkm" if table_name == "multi_species_rpkm" else (
            "fpkm" if table_name.endswith("_fpkm") else "count"
        )

        normalized = re.sub(
            r"^\s*SELECT\s+.+?\s+FROM\s+",
            f"SELECT gene_id, `{sample_column}` AS {metric} FROM ",
            normalized,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )

        normalized = re.sub(
            r"\s+(?:AND|OR)\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?sample_id\s*=\s*'[A-Za-z0-9_]+'",
            "",
            normalized,
            flags=re.IGNORECASE,
        )

    # Hormone identity columns have language-specific meanings.
    if "salvia_hormone_profilling" in normalized.casefold():
        def hormone_identity(match: re.Match[str]) -> str:
            operator, quoted_value = match.group(1), match.group(2)
            bare_value = quoted_value.strip("%")
            if re.fullmatch(r"[A-Z][A-Z0-9-]{1,7}", bare_value):
                column = "class_name"
            elif bare_value.isascii() and re.search(r"[A-Za-z]", bare_value):
                column = "compounds"
            else:
                column = "substance_name"
            return f"{column} {operator} '{quoted_value}'"


        normalized = re.sub(
            r"\bsubstance_name\s*(=|LIKE)\s*'([^']+)'",
            hormone_identity,
            normalized,
            flags=re.IGNORECASE,
        )


    # Enforce field ownership in the cultivated-metabolomics three-table join.
    if all(table in normalized.casefold() for table in (
        "compound_abundance", "salvia_species", "salvia_compounds"
    )):

        normalized = re.sub(r"\bs\.compound_id\b", "c.compound_id", normalized, flags=re.IGNORECASE)

        normalized = re.sub(r"\bs\.abundance\b", "a.abundance", normalized, flags=re.IGNORECASE)

        normalized = re.sub(r"\bs\.name\s+AS\s+compound_name\b", "c.name AS name", normalized, flags=re.IGNORECASE)

        normalized = re.sub(r"\bs\.name\b", "c.name", normalized, flags=re.IGNORECASE)
        requested_sample = re.search(r"\b(S\d{3,})\b", question, re.IGNORECASE)
        if requested_sample and not re.search(r"\bs\.sample_id\s*=", normalized, re.IGNORECASE):
            normalized = normalized.rstrip().rstrip(";")
            conjunction = " AND " if re.search(r"\bWHERE\b", normalized, re.IGNORECASE) else " WHERE "
            sample_id = requested_sample.group(1).upper()
            normalized += f"{conjunction}s.sample_id = '{sample_id}'"

    return normalized


class SalviaComponent:
    def __init__(self):
        self.llm = ChatOllama(
            model=os.getenv("SALVIACHAT_SQL_MODEL", os.getenv("SALVIACHAT_MODEL", "xxx")),
            base_url=os.getenv("SALVIACHAT_OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("SALVIACHAT_TEMPERATURE", "0")),
            num_ctx=int(os.getenv("SALVIACHAT_NUM_CTX", "16000")),
            num_predict=int(os.getenv("SALVIACHAT_SQL_NUM_PREDICT", "512")),
            keep_alive=os.getenv("SALVIACHAT_KEEP_ALIVE", "24h"),
            format="json",
        )
        # self.llm = ChatOpenAI(
        #     model="deepseek-chat", # 
        #     api_key="YOUR_API_KEY",
        #     base_url="https://api.deepseek.com",
        #     temperature=0.3    
        # )
        self.db_config = {
            "host": os.getenv("SALVIACHAT_DB_HOST", "127.0.0.1"),
            "port": int(os.getenv("SALVIACHAT_DB_PORT", "3306")),
            "user": os.getenv("SALVIACHAT_DB_USER", "salviachat"),
            "password": os.getenv("SALVIACHAT_DB_PASSWORD", ""),
            "database": os.getenv("SALVIACHAT_DB_NAME", "salvia")
        }

    def _get_table_schema(self):
        """
        定义数据库表结构，作为 Prompt 传给大模型。
        请根据你实际的数据库表修改这里。
        """
        return """
        == DATABASE NAMING CONVENTION ==
        Our database contains Multi-Omics data for Salvia species. You MUST use the correct naming convention depending on the task.

        0. [TEXT IDENTIFIER JOINS] Some legacy tables use different MySQL
           utf8mb4 collations. For exact textual identifiers such as gene_id,
           sample_id, library_name, compound_id, and taxon, always write
           cross-table joins as `ON BINARY a.identifier = BINARY b.identifier`.
           This rule applies to identifier joins, not numeric comparisons.

        1. [TABLE ROUTING] Transcriptomics & Expression Table Mappings:
           Use these short Latin parts ONLY when constructing dynamic table names (e.g., salvia_{species}_tissue_expression_fpkm).
           - "Salvia miltiorrhiza" (Danshen/丹参/Bunge) -> 'miltiorrhiza'
           - "Salvia hispanica" (Chia/奇亚籽) -> 'hispanica'
           - "Salvia splendens" (一串红) -> 'splendens'
           - "Salvia plebeia" (荔枝草) -> 'plebeia'
           - "Salvia yangii" -> 'yangii'
           - "Salvia castanea" -> 'castanea'
           - "Salvia dorisiana" -> 'dorisiana'
           - "Salvia abrotanoides" -> 'abrotanoides'
           - "Salvia substolonifera" -> 'substolonifera'
           
        2. [COLUMN FILTERING] Genomics Exact Mappings (CRITICAL):
           When querying tables that have a `species_name` column (like `genes`, `chromosome_sequence_chunk`), YOU MUST NEVER use LIKE with short names. You MUST map the user's input to these EXACT strings for the `WHERE` clause:
           - "Danshen", "Bunge", "Shh" -> `WHERE species_name = 'Salvia miltiorrhiza--DSS3'` OR `'Salvia miltiorrhiza--Bunge'` OR `'Salvia miltiorrhiza--shh'` (Depending on specific context, default to DSS3 if not specified)
           - "bowleyana" -> `WHERE species_name = 'Salvia_bowleyana'`
           - "hispanica", "奇亚籽" -> `WHERE species_name = 'Salvia hispanica'`
           - "splendens", "一串红" -> `WHERE species_name = 'Salvia splendens'`
           - "divinorum" -> `WHERE species_name = 'Salvia divinorum'`
           - "sami-il01" -> `WHERE species_name = 'Salvia_miltiorrhiza_sami-il01'`
           
        2. {experiment} & {unit} Mappings:
           - Tissue Expression (FPKM) -> '_tissue_expression_fpkm'
           - Tissue Expression (Count) -> '_tissue_expression_count'
           - Abiotic Stress (FPKM) -> '_abstress_fpkm'
           - Abiotic Stress (Count) -> '_abstress_count'
           
        == TABLE STRUCTURE EXAMPLES ==
        
        [Type A] Gene Info Table:
        Table: genes
        Columns: gene_id, species_name, chromosome, start_bp, end_bp, strand, assembly_name, track_id, symbol, description, aliases
        
        [Type B] Expression Tables (Dynamic Columns):
        Tables like: 'salvia_miltiorrhiza_tissue_expression_fpkm', 'salvia_hispanica_tissue_expression_count' ,'salvia_hispanica_abstress_count'...
        Columns: gene_id, and MANY data columns.
        **WARNING**: These tables **DO NOT** contain RPKM data.
        **IMPORTANT COLUMN NAMING REALITY**:
        - The columns are NOT named simply like 'root' or 'flower'.
        - They look like: 'The_bud_stage_of_Flowering_flower_1', 'AGE_2month_stem_3', 'flowering_stage__at_the_middle_of_vegetation_season_leaf_AB_1'.
        - Columns contain: Stage + Tissue + Replicate.
        
        [Type C] RPKM DATA TABLE (Global Table):
        Table: `multi_species_rpkm`
        **CONTENT**: Contains **RPKM** values for ALL species.
        **CRITICAL RULE**: 
           - IF the user asks for **"RPKM"**, YOU MUST USE THIS TABLE.
           - DO NOT use `salvia_{species}_...` tables for RPKM queries.
        Columns: gene_id, [Species_Replicate Columns like 'Salvia_miltiorrhiza_1', 'Salvia_baimaensis']
        Strategy: ALWAYS use `SELECT *`.
        
        [Type D] Abiotic Stress / Treatment:
        Pattern: salvia_{species}_abstress_{unit} (e.g. salvia_miltiorrhiza_abstress_count)
        Columns: Complex names containing Condition + Time + Tissue.
        Example Columns: 'AGE_1year_Light_0h_leaf_1', 'AGE_25days_NAA_hairy_root_1', etc.
        Strategy: ALWAYS use SELECT * to get all time points.

        [Type E] Sequences:
        Table: gene_sequences
        Columns: 
           - gene_id
           - cds (Coding DNA Sequence, nucleotides ATCG...)
           - rna (Transcript Sequence)
           - protein (Amino Acid Sequence, letters MKLV...)
        **IMPORTANT FOR MULTI-SPECIES**: 
        The `gene_sequences` table does NOT have a species column. To return the species name alongside the sequence, you MUST JOIN it with the `genes` table.
        Example: `... FROM gene_sequences s JOIN genes g ON s.gene_id = g.gene_id ...`
        
        [Type F] Chromosome Sequences (Chunked Storage):
        Table: chromosome_sequence_chunk
        Columns: 
           - species_name (e.g., 'Salvia miltiorrhiza--DSS3')
           - chromosome (e.g., 'GWHAOSJ00000544', 'GWHAOSJ00000545')
           - chunk_index (INT, e.g., 1, 2, 3...)
           - chunk_start (INT, e.g., 1, 20001, 40001)
           - chunk_end (INT, e.g., 40000, 60000, 80000)
           - sequence_chunk (TEXT, The actual DNA string for this chunk)

        [Type G] Phenotype Data (Root Traits):
        Table: root_phenotype_data
        Columns: 
           - id (bigint)
           - sample_id (varchar)
           - dag_stage (int)  -- Days After Germination
           - replicate (int)
           - total_root_length (double)  -- Unit: cm (总根长)
           - total_root_surface_area (double) -- Unit: cm²
           - average_root_diameter (double) -- Unit: mm
           - total_root_volume (double) -- Unit: cm³
           - number_of_root_tips (int) -- Count
           - number_of_root_forks (int) -- Count
           - number_of_crossings (int) -- Count
           - fresh_weight (double) -- Unit: g (鲜重)
           - root_length (double) -- Unit: cm (主根长)
           - primary_lateral_root_number (int)
           - primary_lateral_root_diameter (double) -- Unit: cm
           - secondary_lateral_root_number (int)
           - secondary_lateral_root_diameter (double) -- Unit: cm
        **CRITICAL Phenotype Vocabulary & Unit Mapping:**
        1. **DISTINGUISH ROOT LENGTHS**: 
           - "Total Root Length" or "总根长" -> `total_root_length` (cm)
           - "Root Length" or "主根长" -> `root_length` (cm)
           - If the user asks for "all root lengths" or just "root length" without specifying, YOU MUST SELECT BOTH `root_length` AND `total_root_length`. Do NOT confuse them.
        2. **UNITS MATCHING**: When returning the final SQL, you MUST know that weight is in grams (g), lengths are in cm (except diameter in mm), and volume is in cm³.
        3. **NO SPECIES COLUMN**: This table ONLY contains root phenotype data for Salvia miltiorrhiza. It DOES NOT have a `species_name` or `tissue` column. NEVER use `WHERE species_name = ...` on this table.
        [Type H] Metabolite Info & Abundance (代谢组数据):
        Table: salvia_metabolite_info
        Columns: 
            - compound_id (varchar) -- 代谢物唯一标识
            - pathway (varchar) -- 代谢通路
            - name (varchar) -- 英文名称
            - chinese_name (varchar) -- 中文名称
            - formula (varchar) -- 分子式
            - mz (double) -- 质荷比
            - dag30, dag60, dag90, dag240, dag270, dag390 (double) -- 不同生长阶段（Days After Germination）的代谢物相对丰度/含量数据
        **CRITICAL Metabolite Logic**:
        1. If user asks for "metabolite", "compound", "pathway", "mz", or "abundance", search this table.
        2. Column mapping for time points: "dag30" means "30 days after germination", etc.
        3. Select only requested metadata/DAG fields plus an identifying name or ID.
           Use `SELECT *` only when the user explicitly asks for every available field.

        [Type I] Cultivated Species Metabolomics Analysis:
        Tables: 
        1. salvia_species (Columns: sample_id, chinese_name, scientific_name)
        2. salvia_compounds (Columns: compound_id, name, type)
        3. compound_abundance (Columns: compound_id, sample_id, abundance)
        CRITICAL RULE FOR CULTIVATED SPECIES: 
        To get the abundance of a compound in a specific species, you MUST JOIN all three tables using `sample_id` and `compound_id`.
        Example: SELECT s.scientific_name, s.chinese_name, c.name as compound_name, a.abundance FROM compound_abundance a JOIN salvia_species s ON a.sample_id = s.sample_id JOIN salvia_compounds c ON a.compound_id = c.compound_id

        [Type J] Hormone Profiling Atlas:
        Table: salvia_hormone_profilling
        Columns: 
            - compounds (varchar)
            - substance_name (varchar, e.g., 'IAA', 'ABA')
            - class_name (varchar)
            - formula (varchar)
            - dag30_1, dag30_2, dag30_3, dag60_1 ... dag390_3 (double) -- These are abundance values at different Days After Germination (DAG). The _1, _2, _3 are replicates.
        CRITICAL RULE FOR HORMONES:
        Select the requested stage replicate columns and aggregate them when a stage-level value is requested.
        Use `SELECT *` only for a complete hormone profile across all stages. Use `LIKE` for substance_name.
        Never use a direct `dag60`-style column in this table; it only has replicate-suffixed DAG columns.

        [Type K] Resequencing Data Statistics:
        Table: data_statistics
        Columns: library_name, sra_run, instrument, library_layout, total_bases, avg_spot_length, number_of_reads, total_spots, gc_content
        Description: Contains sequencing quality and basic stats for each library.

        [Type L] Population Info:
        Table: population_info
        Columns: library_name, datastore_provider, organism_name, cultivar, biosample, origin
        Description: Contains the sample source, origin, and cultivar metadata.
        CRITICAL RULE FOR POPULATION & STATS:
        To query both sequencing stats (like GC content, reads) and population metadata (like origin, cultivar), you MUST JOIN `data_statistics` and `population_info` on the `library_name` column.
        Example: SELECT p.cultivar, p.origin, d.total_bases, d.gc_content FROM population_info p JOIN data_statistics d ON p.library_name = d.library_name

        [Type M] Taxon Metadata:
        Table: taxon_metadata
        Columns: taxon, acronym, family, subfamily, tribe_subtribe, genome_size_gb, sampling_date, reproductive_stage, individuals_sampled, voucher, raw_reads, clean_reads, no_rep_transcripts, longest_transcript
        Description: Contains macro-level taxonomy information, genome size, and transcript assembly stats for different taxons.

        [Type N] Gene Variant Summary (SNP & Indel Counts):
        Table: salvia_db_gene_summary
        Columns: gene_id, chrom, start_pos, end_pos, snp_count, indel_count, impact_high, impact_moderate, impact_low, missense_variant, synonymous_variant, intron_variant, etc.
        Description: Use this table when the user asks "How many SNPs/Indels are in gene X?" or "How many high impact variants does this gene have?".

        [Type O] Variant Details (Detailed Mutation Info):
        Table: salvia_db_variant_details
        Columns: var_id, gene_id, chrom, pos, ref_base, alt_base, consequence, impact, hgvs_c, hgvs_p, af (Allele Frequency), ref_ref_count, het_count, alt_alt_count
        Description: Use this when the user asks for detailed information about mutations/variants in a gene, such as "What are the specific missense variants in gene X?". 

        [Type P] Variant Dictionary:
        Table: salvia_variant_dictionary
        Columns: short_var_id, full_var_id

        [Type Q] Gene Structural Features (Exons, CDS, UTRs):
        Table: salvia_gene_feature
        Columns: gene_id, feature_type (values: 'CDS', 'exon', 'five_prime_UTR', 'three_prime_UTR'), start_bp, end_bp
        CRITICAL RULE FOR COUNTING GENE FEATURES:
        Because physical coordinates might overlap, you MUST use `COUNT(DISTINCT CONCAT(start_bp, '_', end_bp))` for accurate counting.
        - Exon count: `COUNT(DISTINCT CASE WHEN feature_type = 'exon' THEN CONCAT(start_bp, '_', end_bp) ELSE NULL END)`
        - Intron count is (Exon Count - 1). If exon count <= 1, intron count is 0.
        
        **CRITICAL LOGIC FOR SEQUENCES**:
        - The genome is huge, so it's split into chunks.
        - To find a sequence for interval [UserStart, UserEnd], you must find ALL chunks that overlap this interval.
        - **Overlap Formula**: `chunk_start <= UserEnd AND chunk_end >= UserStart`.
        """
        

    def generate_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: 理解用户问题，生成 SQL"""
        metadata = state.get("metadata", {})
        messages = state["messages"]
        
        # 获取用户问题
        user_question = [msg for msg in messages if isinstance(msg, HumanMessage)]
        
        if not user_question:
            return {"status": "error", "error": "No user question found"}
            
        original_question = user_question[-1].content
        # 把 Router 查出来的官方 ID 合并给大模型看
        resolved_messages = [msg for msg in messages if isinstance(msg, SystemMessage) and msg.additional_kwargs.get("type") == "entity_linking_resolved"]
        
        if resolved_messages:
            # 如果有图谱消歧的结果，强行拼接到大模型要看的问题里
            linker_hint = resolved_messages[-1].content
            original_question = f"{original_question}\n\n{linker_hint}"
            logger.info("Injected resolved ID into Coder prompt.")
        logger.info(f"Generating SQL for: {original_question}")
        eval_result = metadata.get("eval_result", {})
        correction_feedback = eval_result.get("reason", "") if eval_result.get("next_step") == "router" else ""
        database_errors = [
            msg.content for msg in messages
            if isinstance(msg, AIMessage)
            and msg.additional_kwargs.get("type") == "salvia_error"
        ]
        database_error_feedback = database_errors[-1] if database_errors else ""
        if database_error_feedback:
            correction_feedback = (
                f"{correction_feedback}\nPrevious SQL failed with: {database_error_feedback}. "
                "Do not repeat the failed SQL; correct its table, column, join, grouping, or collation."
            ).strip()
        if correction_feedback:
            logger.info(f"Applying evaluator correction feedback: {correction_feedback}")

        # 构建 Prompt
        system_prompt = f"""
You are an expert Data Engineer for a Salvia Multi-Omics Database.
Your task is to translate natural language questions into MySQL queries.
DATABASE SCHEMA:
{self._get_table_schema()}

PREVIOUS EVALUATOR CORRECTION:
{correction_feedback if correction_feedback else "No correction; this is the first attempt."}

OUTPUT-COLUMN RULES (HIGHEST PRIORITY):
1. SELECT the fields explicitly requested by the user. Do not replace requested
   annotation fields with coordinates or sequences.
2. For gene annotation/function questions, query `genes` and select `gene_id`,
   `species_name`, `description`, `symbol`, and `aliases` as requested. Do not
   join `gene_sequences` unless a sequence was requested.
3. Use this canonical alias contract for derived values:
   - `AVG(...) AS mean_value`
   - `MIN(...) AS min_value`
   - `MAX(...) AS max_value`
   - subtraction as `difference`
   - ratio as `fold_change`
   - `COUNT(*) AS sample_count` when the user requests a sample count
   Do not encode the trait name or unit inside a derived-value alias.
4. If the user explicitly requests a unit, include the known schema unit as a
   literal `unit` column, for example `'cm' AS unit`, `'cm^2' AS unit`,
   `'cm^3' AS unit`, `'mm' AS unit`, `'g' AS unit`, or `'count' AS unit`.
5. Return exactly the requested entity scope. Do not add unrelated rows, stages,
   species, fields, or interpretations. Add an identifier only when necessary
   to distinguish returned records.
6. Use `SELECT *` only for genuinely dynamic expression/RPKM columns or when the
   user explicitly requests all fields. Otherwise project only requested data
   plus the entity identifier needed to interpret it.
7. If correction feedback lists missing fields, the new SELECT clause MUST include those fields.
8. Never mix DAG schemas: `salvia_metabolite_info` has direct columns `dag30`,
   `dag60`, `dag90`, `dag240`, `dag270`, `dag390` with NO replicate suffix.
   Only `salvia_hormone_profilling` uses `dag60_1`, `dag60_2`, `dag60_3`, etc.
   The selected table determines which convention is valid.
9. Expression and RPKM tables are WIDE tables and contain no `sample_id`
   column. An exact sample name such as `flowering_corolla_2` is a physical
   column: select it with backticks and filter only by `gene_id`. Never write
   `WHERE sample_id = ...` for an expression or RPKM table.
10. In `salvia_hormone_profilling`, `compounds` stores the English full name
    (for example Abscisic acid), `substance_name` stores the Chinese name, and
    `class_name` stores abbreviations such as ABA. Match the field to the
    language and form of the user's identifier.
11. Treat an identifier containing `&` as one exact identifier. Never split
    `EVM0000525&EVM0001279` into two values.
12. Cultivated-metabolomics aliases have fixed field ownership:
    `a(compound_id,sample_id,abundance)`,
    `c(compound_id,name,type)`, and
    `s(sample_id,chinese_name,scientific_name)`.

CRITICAL RULE FOR GENE IDs:
If the User Question contains a <RESOLVED_ID> block, you MUST extract the official ID inside it and use it in your SQL `WHERE gene_id = '...'` or `WHERE gene_id IN (...)` clause. Do NOT use the user's original short alias (like ABCC1).

CRITICAL RULES FOR CROSS-SPECIES QUERIES (NO SPECIES SPECIFIED):
1. If the user does not specify a species (or wants all species), you MUST NOT restrict the query to a single species in the WHERE clause.
2. Your `SELECT` statement MUST explicitly include the species name and gene_id so the user knows which sequence belongs to which species.
3. For sequences (Type E), you MUST use a JOIN to get the species name. 
   - Correct template: `SELECT g.species_name, s.gene_id, s.cds FROM gene_sequences s JOIN genes g ON s.gene_id = g.gene_id WHERE s.gene_id IN (...)`

CRITICAL RULES FOR CROSS-OMICS QUESTIONS:
1. Never invent a join column between unrelated omics tables:
   - `salvia_metabolite_info` has no `sample_id`.
   - `salvia_hormone_profilling` has no `sample_id`.
   - `taxon_metadata` has no `library_name`.
   - `root_phenotype_data` has `root_length`, not `primary_root_length`.
2. When facts require independent tables, return independent SELECT statements as
   `{{"sql": ["SELECT ...", "SELECT ..."]}}` in execution order. Do not force
   unrelated tables into one JOIN, UNION, or IN subquery.
3. Gene/expression and gene/sequence tables use different collations. Prefer
   independent SELECTs; any unavoidable column join must COLLATE both operands identically.
4. Every aggregate query with a non-aggregated selected column must GROUP BY that column.
5. Never use `LIMIT` inside an `IN`, `ALL`, `ANY`, or `SOME` subquery.
6. Query phenotype and metabolite separately for a shared DAG-stage comparison.
7. Query phenotype and hormone separately; select `root_length` for primary root length.
8. For population + sequencing + taxon metadata, issue three queries:
   - population by `population_info.library_name`, selecting `library_name`, `organism_name`, `cultivar`, and `origin`;
   - sequencing by `data_statistics.library_name`, selecting `library_name`, `sra_run`, `number_of_reads`, `total_bases`, and `gc_content`;
   - taxon by `taxon_metadata.taxon` using the exact organism name. If only a
     library is supplied, use `WHERE taxon = (SELECT organism_name FROM
     population_info WHERE library_name = 'ACJ_AA')`. Never broaden this to
     `LIKE '%Salvia%'` and never write `t.library_name`.

CRITICAL RULE FOR FULL INFO:
- 当用户要求 "all information", "complete data" 或 "全部信息" 时：
- Gene metadata must include `g.description`, `g.symbol`, and `g.aliases` in addition to location fields.
- Join `gene_sequences` and select CDS/RNA/protein only when sequence data is explicitly requested.
- Never claim to retrieve "all information" while omitting description, symbol, or aliases.

DECISION LOGIC (Follow Strict Priority):
1. **Check for "RPKM" Keyword**:
   - Does the user ask for "RPKM"? 
   - **YES** -> USE table `multi_species_rpkm`. 
     - *Ignore* the specific species table naming convention.
     - *Example*: "RPKM in Danshen" -> `SELECT * FROM multi_species_rpkm ...`
   - **NO** -> Proceed to step 2.

2. **Specific Tissue Expression**? 
   - IF user asks for "root", "leaf", "tissue expression" for a SPECIFIC species.
   - USE: `salvia_{{species}}_tissue_expression_fpkm` (or `_count`).
   
3. **Gene Info**?
   - USE: `genes`.

4. **Abiotic Stress / Treatment**?
   - IF user asks for "stress", "light", "heat", "cold", "drought", "treatment".
   - USE: `salvia_{{species}}_abstress_{{unit}}`.
   - STRATEGY: SELECT *.

5. **Sequences** (CDS, Protein, RNA)?
   - USE: `gene_sequences`.
   - Columns: Select the specific type requested (e.g., `SELECT protein`).
   - If user asks for "sequence" generally, `SELECT *`.

6. **Genome Sequence (Interval Search)**?
   - User asks: "What is the sequence of Chr1 from 100 to 200?", "Show me the DNA string...".
   - **Action**: Query `chromosome_sequence_chunk`.
   - **SQL Logic (Overlap)**: 
     `SELECT * FROM chromosome_sequence_chunk WHERE species_name = '...' AND chromosome = '...' AND chunk_start <= [UserEnd] AND chunk_end >= [UserStart] ORDER BY chunk_start`
   - *Note*: Always ORDER BY chunk_start so we can stitch them correctly later.

7. **Gene Density / Distribution in Region or Species**?
           - User asks: "gene density", "Top 8 gene density in bunge", "gene density of Chr01 in Salvia miltiorrhiza--Bunge".
           - **Action**: Query `genes` table.
           - **CRITICAL DEFINITION**: If the user asks for "Top 8" or "Top N" gene density in a species, they mean finding the Top N **CHROMOSOMES** with the most genes. THEY DO NOT MEAN dividing a single chromosome into regions.
           - **CRITICAL SQL Logic**: 
             1. ALWAYS `SELECT gene_id, species_name, chromosome, start_bp`. (You MUST include `gene_id` so the frontend data table can display the specific gene names).
             2. YOU MUST NEVER USE `COUNT(*)` OR `LIMIT` FOR GENE DENSITY QUERIES. You must retrieve coordinates for ALL genes so the backend can group and rank them.
             3. **Species Matching**: If a user mentions a species (e.g., "bunge", "hispanica"), you MUST look up its exact name in the `[COLUMN FILTERING] Genomics Exact Mappings` dictionary. Example: Use `WHERE species_name = 'Salvia miltiorrhiza--Bunge'`. NEVER use `LIKE '%bunge%'`.

8. **Phenotype / Root Traits**?
   - User asks: "root length", "surface area", "fresh weight", "root tips", "phenotype".
   - **Action**: Query `root_phenotype_data`.
   - **SQL Logic**: 
     - If user asks for specific sample (e.g., "for sample S1"), use `WHERE sample_id = 'S1'`.
     - If user asks for specific stage (e.g., "at DAG 14"), use `WHERE dag_stage = 14`.
     - Always SELECT `sample_id`, `dag_stage`, and the specifically requested trait columns. If asked for "all phenotypes", SELECT *.
     - **CRITICAL**: If the user explicitly mentions "Salvia miltiorrhiza" or "root" in a phenotype query, IGNORE the species mapping rule. DO NOT add `species_name` to the WHERE clause for this table, or it will crash.
9. **Metabolite / Compound / Metabolomics**?
    - User asks: "metabolite abundance", "chinese name of compound", "pathway", "mz of compound".
    - **Action**: Query `salvia_metabolite_info`.
    - **SQL Logic**:
    - If searching by name: `WHERE name LIKE '%...%' OR chinese_name LIKE '%...%'`
    - If searching by ID: `WHERE compound_id = '...'`
    - A named compound compared across DAG stages defaults to this table, even if the word "metabolite" is omitted.
    - Rosmarinic acid, Salvianolic acid, and Tanshinone use its direct `dag30` through `dag390` columns.
    - DAG-stage wording alone is not evidence that the hormone table should be used.
    - Select the requested metadata and DAG columns explicitly. Use `SELECT *` only when the user asks for all available metabolite fields.
    - Alias selected DAG values canonically as `dag30`, `dag60`, `dag90`, `dag240`, `dag270`, and `dag390`.

10. **Cultivated Species Metabolomics**?
    - User asks: "abundance of compound X in cultivated species", "different salvia species".
    - **Action**: JOIN `compound_abundance a`, `salvia_species s`, `salvia_compounds c`.
    - **SQL Logic**: filter the compound with `c.name`. When the question
      supplies a sample ID such as S0123, also use exact
      `s.sample_id = 'S0123'`; do not guess from scientific_name alone.

11. **Hormone Profiling / Plant Hormones**?
    - Use this table only when the question explicitly identifies a hormone or hormone profiling; do not route an arbitrary named metabolite here merely because DAG stages are mentioned.
    - User asks: "hormone", "IAA", "ABA", "JA", "substance_name", "hormone atlas".
    - **Action**: Query `salvia_hormone_profilling`.
    - **SQL Logic**: English full name -> `compounds`; Chinese name ->
      `substance_name`; abbreviation such as ABA/IAA -> `class_name`.
    - Stage values are replicate columns such as `dag60_1`, `dag60_2`, and `dag60_3`; columns named only `dag60` or `dag240` do not exist.
    - When one stage is requested, return its replicate mean, for example `(dag60_1 + dag60_2 + dag60_3) / 3 AS dag60_mean`.
    - For a difference between stages, subtract the two replicate means and use a semantic alias such as `difference`.
    - Never invent a non-replicate DAG column.

12. **Resequencing & Population Info**?
    - User asks: "library layout", "GC content", "reads", "cultivar origin", "biosample".
    - **Action**: Query `data_statistics` or `population_info`. 
    - **SQL Logic**: If the query needs both origin/cultivar AND stats (like reads/GC), ALWAYS JOIN them: `... FROM population_info p JOIN data_statistics d ON p.library_name = d.library_name WHERE ...`

13. **Taxon Metadata**?
    - User asks: "genome size", "taxonomy", "family", "subfamily", "transcript stats".
    - **Action**: Query `taxon_metadata`.
    - **SQL Logic**: `WHERE taxon LIKE '%...%' OR acronym = '...'`
14. **Gene Variant/Mutation Summary & Impact Distribution**?
           - User asks: "How many SNPs/Indels", "variant impact distribution", "mutation effects summary", "high/moderate/low/modifier counts for a gene".
           - **Action**: Query `salvia_db_gene_summary`.
           - **CRITICAL RULE**: For overall gene mutation distribution or pie charts, ALWAYS query this table's pre-computed columns. DO NOT use `GROUP BY` on `salvia_db_variant_details` because it causes transcript-level inflation, especially for `MODIFIER`.
           - **SQL Logic**: `SELECT gene_id, snp_count, indel_count, impact_high, impact_moderate, impact_low, impact_modifier FROM salvia_db_gene_summary WHERE gene_id = '...'`

15. **Detailed Variant Information (AF, Consequence)**?
    - User asks: "Show me the variant details for gene X", "What are the high impact variants in gene Y?".
    - **Action**: Query `salvia_db_variant_details`.
    - **SQL Logic**: `SELECT var_id, pos, ref_base, alt_base, consequence, impact, af FROM salvia_db_variant_details WHERE gene_id = '...'`

16. **Gene Structure (Exons, Introns, CDS Counts)**?
    - User asks: "How many exons does gene X have?", "Calculate the introns and CDS".
    - **Action**: Query `salvia_gene_feature`.
    - **SQL Logic**: YOU MUST USE THE COMPLEX `COUNT(DISTINCT)` LOGIC to prevent duplicate counting. Group by gene_id.
    
CRITICAL STRATEGY FOR TISSUE EXPRESSION (Type B):
1. **THE "SELECT *" RULE**: Because column names are complex (e.g., 'The_bud_stage_of_Flowering_flower_1'), **NEVER try to guess specific columns** like `SELECT flower` or `SELECT stem`.
2. **ALWAYS USE `SELECT *`**.
   - Why? We need to retrieve ALL columns (all tissues/stages) and let the downstream analyzer filter them.
   - WRONG: `SELECT flower FROM salvia_miltiorrhiza_tissue_expression_fpkm ...` (Column 'flower' does not exist!)
   - CORRECT: `SELECT * FROM salvia_miltiorrhiza_tissue_expression_fpkm WHERE gene_id = '...'`

CRITICAL RULES FOR 'multi_species_rpkm':
1. **THE "SELECT *" RULE**: Because column names are irregular (some have '_1', some don't), **YOU MUST ALWAYS USE `SELECT *`** for this table.
   - WRONG: `SELECT Salvia_baimaensis FROM multi_species_rpkm ...` (Risk: Column might be named Salvia_baimaensis_1)
   - WRONG: `SELECT Salvia_miltiorrhiza FROM ...` (Risk: Columns are _1, _2)
   - CORRECT: `SELECT * FROM multi_species_rpkm WHERE gene_id = '...'`

RULES:
- Always try to find the `gene_id` first. 
- For `multi_species_rpkm`, if the user asks for "all species", use `SELECT *`.

FEW-SHOT EXAMPLES:

User: "返回 ABCC1 的 CDS 序列 [假设 <RESOLVED_ID> 解析为了 'Sm_001', 'Sh_001']"
Reasoning: Sequence query. Target=cds. NO species specified, which means this is a CROSS-SPECIES query. I must JOIN with `genes` table to include `species_name`.
SQL: SELECT g.species_name, s.gene_id, s.cds FROM gene_sequences s JOIN genes g ON s.gene_id = g.gene_id WHERE s.gene_id IN ('Sm_001', 'Sh_001');

User: "Get the protein sequence of GWHGAOSJ000002."
Reasoning: Specific ID provided. Target=protein. Direct query.
SQL: SELECT protein FROM gene_sequences WHERE gene_id = 'GWHGAOSJ000002';

User: "Show me the expression of 'GWHGAOSJ000003' in Salvia miltiorrhiza flower."
Reasoning: User wants 'flower'. The table has columns like 'The_bud_stage_of_Flowering_flower_1'. I cannot guess the exact name. I must Select All.
SQL: SELECT * FROM salvia_miltiorrhiza_tissue_expression_fpkm WHERE gene_id = 'GWHGAOSJ000003';

User: "Get stem count data for GWHGAOSJ000003 in Danshen."
Reasoning: Species=miltiorrhiza, Unit=Count. Strategy -> Select All.
SQL: SELECT * FROM salvia_miltiorrhiza_tissue_expression_count WHERE gene_id = 'GWHGAOSJ000003';

User: "Show me the FPKM expression of gene 'GWHGAOSJ000002' in Danshen root."
Reasoning: Tissue columns are dynamic and cannot be guessed. Retrieve the row and let deterministic downstream filtering retain root columns.
SQL: SELECT * FROM salvia_miltiorrhiza_tissue_expression_fpkm WHERE gene_id = 'GWHGAOSJ000002';

User: "Show me the RPKM of 'GWHGAOSJ000003' in Salvia miltiorrhiza."
Reasoning: User asks for **RPKM**. Even though "Salvia miltiorrhiza" is a specific species, RPKM data is ALWAYS in the multi-species table.
SQL: SELECT * FROM multi_species_rpkm WHERE gene_id = 'GWHGAOSJ000003';

User: "What is the RPKM of gene GWHGAOSJ000003 in Salvia baimaensis?"
Reasoning: Keyword "RPKM". Use multi_species_rpkm.
SQL: SELECT * FROM multi_species_rpkm WHERE gene_id = 'GWHGAOSJ000003';

User: "Get the protein sequence of GWHGAOSJ000002."
Reasoning: Sequence query. Target=protein.
SQL: SELECT protein FROM gene_sequences WHERE gene_id = 'GWHGAOSJ000002';

User: "Show me the CDS and RNA for GWHGAOSJ000003."
Reasoning: Sequence query. Targets=cds, rna. ID used -> Direct query.
SQL: SELECT cds, rna FROM gene_sequences WHERE gene_id = 'GWHGAOSJ000003';

User: "Get the sequence of Salvia miltiorrhiza--DSS3 Chr1 from base 5000 to 7000."
Reasoning: Interval overlap query on chunked table.
SQL: SELECT * FROM chromosome_sequence_chunk WHERE species_name = 'Salvia miltiorrhiza--DSS3' AND chromosome = 'Chr1' AND chunk_start <= 5000 AND chunk_end >= 7000 ORDER BY chunk_start;

User: "Help me output all the information of the gene ABCC1"
Reasoning: "All information" means I need both metadata from 'genes' and sequences from 'gene_sequences'.
SQL: SELECT g.species_name, g.gene_id, g.chromosome, g.start_bp, g.end_bp, g.strand, g.description, g.symbol, g.aliases, s.cds, s.rna, s.protein FROM genes g LEFT JOIN gene_sequences s ON g.gene_id = s.gene_id WHERE g.gene_id IN ('ID1', 'ID2');

User: "What is the gene density of chromosome GWHAOSJ00000544?"
Reasoning: Keyword "gene density". No species specified. I must SELECT gene_id, species_name, chromosome, start_bp. I must NEVER use COUNT(*).
SQL: SELECT gene_id, species_name, chromosome, start_bp FROM genes WHERE chromosome = 'GWHAOSJ00000544';

User: "Show me the abundance of Tanshinone IIA across different cultivated Salvia species."
Reasoning: Keyword "cultivated species" and "abundance". Needs 3-table JOIN.
SQL: SELECT s.scientific_name, s.chinese_name, c.name as compound_name, a.abundance FROM compound_abundance a JOIN salvia_species s ON a.sample_id = s.sample_id JOIN salvia_compounds c ON a.compound_id = c.compound_id WHERE c.name LIKE '%Tanshinone IIA%';

User: "What is the hormone profile of IAA?"
Reasoning: Keyword "hormone profile" and "IAA". Use hormone table and SELECT *.
SQL: SELECT * FROM salvia_hormone_profilling WHERE substance_name LIKE '%IAA%' OR compounds LIKE '%IAA%';

User: "Compare Rosmarinic acid abundance at DAG60 and DAG240 and calculate the latter minus the former."
Reasoning: Named metabolite with direct DAG columns. Use salvia_metabolite_info, not the hormone table.
SQL: SELECT name, dag60 AS value_a, dag240 AS value_b, (dag240 - dag60) AS absolute_change FROM salvia_metabolite_info WHERE name LIKE '%Rosmarinic acid%';

User: "What is the GC content and total bases for the library SRR123456?"
Reasoning: Keyword "GC content", "library". Targets table data_statistics.
SQL: SELECT library_name, sra_run, total_bases, gc_content FROM data_statistics WHERE sra_run = 'SRR123456' OR library_name = 'SRR123456';

User: "Show me the GC content and number of reads for cultivar 'Tanshen 1'."
Reasoning: User needs stats (data_statistics) based on a cultivar (population_info). Needs a JOIN on library_name.
SQL: SELECT p.cultivar, p.origin, d.library_name, d.gc_content, d.number_of_reads FROM population_info p JOIN data_statistics d ON p.library_name = d.library_name WHERE p.cultivar LIKE '%Tanshen 1%';

User: "What is the genome size and family of the taxon Salvia miltiorrhiza?"
Reasoning: Keyword "genome size", "family". Targets table taxon_metadata.
SQL: SELECT taxon, family, genome_size_gb FROM taxon_metadata WHERE taxon LIKE '%Salvia miltiorrhiza%';

User: "Compare the genome size Gb and the number of clean reads of Salvia chinensis Benth and Salvia japonica Thunb."
Reasoning: Both `genome_size_gb` and `clean_reads` are columns directly inside the `taxon_metadata` table. NO JOIN is needed.
SQL: SELECT taxon, genome_size_gb, clean_reads FROM taxon_metadata WHERE taxon IN ('Salvia chinensis Benth', 'Salvia japonica Thunb');

User: "How many SNPs and indels are there in gene GWHGAOSJ000003?"
Reasoning: User wants summary counts of mutations. Use salvia_db_gene_summary.
SQL: SELECT gene_id, snp_count, indel_count FROM salvia_db_gene_summary WHERE gene_id = 'GWHGAOSJ000003';

User: "Show me the details of high impact variants in gene GWHGAOSJ000003."
Reasoning: User wants specific details of variants with HIGH impact. Use salvia_db_variant_details.
SQL: SELECT var_id, chrom, pos, ref_base, alt_base, consequence, af FROM salvia_db_variant_details WHERE gene_id = 'GWHGAOSJ000003' AND impact = 'HIGH';

User: "How many exons, CDS, and introns does the gene GWHGAOSJ000003 have?"
Reasoning: User is asking for gene structural features. I must use `salvia_gene_feature` and the complex distinct counting logic provided in the schema.
SQL: SELECT gene_id, COUNT(DISTINCT CASE WHEN feature_type = 'CDS' THEN CONCAT(start_bp, '_', end_bp) ELSE NULL END) AS cdsCount, COUNT(DISTINCT CASE WHEN feature_type = 'exon' THEN CONCAT(start_bp, '_', end_bp) ELSE NULL END) AS exonCount, IF(COUNT(DISTINCT CASE WHEN feature_type = 'exon' THEN CONCAT(start_bp, '_', end_bp) ELSE NULL END) > 1, COUNT(DISTINCT CASE WHEN feature_type = 'exon' THEN CONCAT(start_bp, '_', end_bp) ELSE NULL END) - 1, 0) AS intronCount FROM salvia_gene_feature WHERE gene_id = 'GWHGAOSJ000003' GROUP BY gene_id;

User: "What is the variant impact distribution for gene GWHGAOSJ000003? Help me plot a pie chart to see the mutation effects."
Reasoning: User wants the variant impact distribution (mutation effects summary) for a single gene. The accurate gene-level deduplicated counts are stored in `salvia_db_gene_summary`. I must query the summary columns directly and NOT use GROUP BY on the details table. I must also include snp_count and indel_count for comprehensive visualization.
SQL: SELECT gene_id, snp_count, indel_count, impact_high, impact_moderate, impact_low, impact_modifier FROM salvia_db_gene_summary WHERE gene_id = 'GWHGAOSJ000003';

User: "What is the Top 8 gene density in bunge?"
Reasoning: Keyword "gene density" and "bunge". "bunge" strictly maps to 'Salvia miltiorrhiza--Bunge' according to the Exact Mappings. I must get all coordinates for this exact species without using LIMIT, so the Python backend can rank the Top 8 chromosomes. I must include gene_id.
SQL: SELECT gene_id, species_name, chromosome, start_bp FROM genes WHERE species_name = 'Salvia miltiorrhiza--Bunge';

User: "What is the gene density of chromosome Chr01 in Salvia miltiorrhiza--Bunge?"
Reasoning: Keyword "gene density". The user specified both chromosome and species. I must use the exact species string 'Salvia miltiorrhiza--Bunge'. I must retrieve gene_id, species_name, chromosome, start_bp.
SQL: SELECT gene_id, species_name, chromosome, start_bp FROM genes WHERE chromosome = 'Chr01' AND species_name = 'Salvia miltiorrhiza--Bunge';

User: "Compare the root fresh weight across all samples at DAG 15 in Salvia miltiorrhiza."
Reasoning: Keyword "fresh weight" and "DAG 15". Targets table `root_phenotype_data`. CRITICAL: Although the user mentions "Salvia miltiorrhiza", this phenotype table DOES NOT have a `species_name` column. I MUST NOT add `species_name` to the WHERE clause.
SQL: SELECT sample_id, dag_stage, fresh_weight, 'g' AS unit FROM root_phenotype_data WHERE dag_stage = 15;

Cross-omics example — gene location plus expression:
Output: {{"sql": ["SELECT gene_id, species_name, chromosome, start_bp, end_bp, strand FROM genes WHERE gene_id = 'GWHGAOSJ000001'", "SELECT gene_id, `AGE_1year_root_1` AS fpkm FROM salvia_miltiorrhiza_tissue_expression_fpkm WHERE gene_id = 'GWHGAOSJ000001'"]}}

Cross-omics example — phenotype plus metabolite:
Output: {{"sql": ["SELECT dag_stage, AVG(fresh_weight) AS mean_value, COUNT(*) AS sample_count, 'g' AS unit FROM root_phenotype_data WHERE dag_stage = 30 GROUP BY dag_stage", "SELECT compound_id, name, dag30 AS abundance FROM salvia_metabolite_info WHERE name LIKE '%Salvianolic acid B%'"]}}

Cross-omics example — phenotype plus hormone replicates:
Output: {{"sql": ["SELECT dag_stage, AVG(root_length) AS mean_value, 'cm' AS unit FROM root_phenotype_data WHERE dag_stage = 90 GROUP BY dag_stage", "SELECT compounds, substance_name, dag90_1, dag90_2, dag90_3 FROM salvia_hormone_profilling WHERE id = 3"]}}

Cross-omics example — population, sequencing, and taxon metadata:
Output: {{"sql": ["SELECT library_name, organism_name, cultivar, origin FROM population_info WHERE library_name = 'ACJ_AJ'", "SELECT library_name, sra_run, number_of_reads, total_bases, gc_content FROM data_statistics WHERE library_name = 'ACJ_AJ'", "SELECT taxon, genome_size_gb, raw_reads, clean_reads FROM taxon_metadata WHERE taxon = 'Salvia hispanica'"]}}

User Question: {original_question}
For one query output: {{"sql": "SELECT ..."}}
For independent cross-omics queries output: {{"sql": ["SELECT ...", "SELECT ..."]}}
Return only valid JSON. Every statement must be a read-only SELECT.
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
            
            content = response.content.replace("```json", "").replace("```", "").strip()
            json_match = re.search(r'({.*?})', content, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group(1))
            else:
                params = json.loads(content)
                
            sql_query = params.get("sql")
            if isinstance(sql_query, str):
                sql_queries = [sql_query.strip()]
            elif isinstance(sql_query, list):
                sql_queries = [str(query).strip() for query in sql_query if str(query).strip()]
            else:
                raise ValueError("The 'sql' field must be a SELECT string or a list of SELECT strings")
            if not sql_queries or any(not re.match(r"^\s*SELECT\b", query, re.IGNORECASE) for query in sql_queries):
                raise ValueError("Only non-empty read-only SELECT statements are allowed")
            if any(";" in query.rstrip().rstrip(";") for query in sql_queries):
                raise ValueError("Each SQL array item must contain exactly one statement")
            sql_queries = [
                _normalize_generated_sql(query, original_question)
                for query in sql_queries
            ]
            sql_query = sql_queries[0] if len(sql_queries) == 1 else sql_queries
            
            return {
                "messages": [
                    AIMessage(
                        content=f"Generated SQL: {sql_query}",
                        additional_kwargs={"type": "salvia_sql_progress", "sql": sql_query}
                    )
                ],
                "next": "execute_sql", # 指向下一个节点
                "metadata": {**metadata, "current_sql": sql_query}
            }

        except Exception as e:
            logger.error(f"SQL Generation Error: {e}")
            return {
                "status": "error", 
                "messages": [AIMessage(content=f"SQL Generation failed: {str(e)}")]
            }

    def execute_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: 执行 SQL 并返回结果"""
        metadata = state.get("metadata", {})
        messages = state["messages"]
        sql_query = metadata.get("current_sql")

        if not sql_query:
            return {"status": "error", "error": "No SQL found to execute"}

        try:
            sql_queries = sql_query if isinstance(sql_query, list) else [sql_query]
            sql_text = "\n".join(sql_queries)
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True) # 让结果以字典形式返回 {col: val}
            result_batches = []
            executed_queries = []
            try:
                for query_index, query in enumerate(sql_queries, 1):
                    logger.info(f"Executing SQL {query_index}/{len(sql_queries)}: {query}")
                    executed_query = query
                    try:
                        cursor.execute(executed_query)
                    except mysql.connector.Error as db_error:
                        if db_error.errno != 1267:
                            raise
                        repaired_query = _repair_collation_equalities(executed_query)
                        if repaired_query == executed_query:
                            raise
                        logger.warning(
                            "MySQL collation mismatch detected; retrying with "
                            "byte-exact identifier comparison: %s",
                            repaired_query,
                        )
                        cursor.execute(repaired_query)
                        executed_query = repaired_query
                    executed_queries.append(executed_query)
                    result_batches.append(cursor.fetchall())
            finally:
                cursor.close()
                conn.close()

            sql_query = executed_queries[0] if len(executed_queries) == 1 else executed_queries
            sql_text = "\n".join(executed_queries)

            if len(result_batches) == 1:
                results = list(result_batches[0])
            else:
                # Independent cross-omics queries must remain separate. Merging
                # one-row batches can overwrite shared keys (for example
                # gene_id) and later column filtering can discard an entire
                # omics section.
                results = []
                for query_index, batch in enumerate(result_batches, 1):
                    for source_row in batch:
                        row = dict(source_row)
                        row["query_index"] = query_index
                        results.append(row)

            current_sql = sql_text.lower()
            
            if "chromosome_sequence_chunk" in current_sql and results:
                try:

                    full_sequence = ""

                    sorted_results = sorted(results, key=lambda x: x['chunk_start'])
                    for chunk in sorted_results:
                        if chunk.get('sequence_chunk'):
                            full_sequence += chunk['sequence_chunk']
                    
                    summary_result = [{
                        "species": sorted_results[0].get('species_name'),
                        "chromosome": sorted_results[0].get('chromosome'),
                        "stitched_sequence_length": len(full_sequence),
                        "stitched_sequence_preview": full_sequence[:50] + "..." + full_sequence[-50:] if len(full_sequence) > 100 else full_sequence,
                        "full_sequence_raw": full_sequence # 这个字段留给 Generator 决定是否展示
                    }]
                    
                    results = summary_result
                    logger.info(f"Stitched {len(sorted_results)} sequence chunks into one string.")
                    
                except Exception as e:
                    logger.error(f"Error stitching sequences: {e}")
            if "root_phenotype_data" in sql_text.lower() and results:
                unit_mapping = {
                    "total_root_length": "total_root_length (cm )",
                    "total_root_surface_area": "total_root_surface_area (cm²)",
                    "average_root_diameter": "average_root_diameter (mm)",
                    "total_root_volume": "total_root_volume (cm³)",
                    "fresh_weight": "fresh_weight (g )",
                    "root_length": "root_length (cm )",
                    "primary_lateral_root_diameter": "primary_lateral_root_diameter (cm)",
                    "secondary_lateral_root_diameter": "secondary_lateral_root_diameter (cm)"
                }
                
                formatted_results = []
                for row in results:
                    new_row = {}
                    for k, v in row.items():
                        new_key = unit_mapping.get(k, k)
                        new_row[new_key] = v
                    formatted_results.append(new_row)
                
                # 覆盖原来的 results
                results = formatted_results
                logger.info("Injected phenotype units into results.")

            if "salvia_metabolite_info" in sql_text.lower() and results:

                metabolite_unit_map = {
                    "dag30": "30 DAG (Abundance)",
                    "dag60": "60 DAG (Abundance)",
                    "dag90": "90 DAG (Abundance)",
                    "dag240": "240 DAG (Abundance)",
                    "dag270": "270 DAG (Abundance)",
                    "dag390": "390 DAG (Abundance)"
                }
                formatted_metabolites = []
                for row in results:
                    new_row = {}
                    for k, v in row.items():
                        new_key = metabolite_unit_map.get(k, k)
                        new_row[new_key] = v
                    formatted_metabolites.append(new_row)
                results = formatted_metabolites
            user_question_text = ""
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    user_question_text = msg.content.lower()

            keywords = ["leaf", "root", "stem", "flower", "calyx", "pistil", "stamen"]
            target_keywords = [k for k in keywords if k in user_question_text]

            if "salvia_hormone_profilling" in sql_text.lower() and results:
                formatted_hormones = []
                for row in results:
                    new_row = {}
                    for k, v in row.items():
                        import re
                        match = re.match(r'dag(\d+)_(\d+)', k.lower())
                        if match:
                            day = match.group(1)
                            rep = match.group(2)
                            new_key = f"{day} DAG hormone abundance (Rep {rep}; unit unavailable)"
                            new_row[new_key] = v
                        else:
                            new_row[k] = v
                    formatted_hormones.append(new_row)
                results = formatted_hormones
                logger.info("Injected hormone profiling headers into results.")
            
            if results and target_keywords and any(len(row) > 10 for row in results):
                filtered_results = []
                for row in results:
                    if len(row) <= 10:
                        filtered_results.append(row)
                        continue
                    new_row = {}
                    for col, val in row.items():
                        col_lower = col.lower()
                        if col in {"query_index", "gene_id"} or "gene_id" in col_lower:
                            new_row[col] = val
                            continue
                        if any(target in col_lower for target in target_keywords):
                            new_row[col] = val
                    filtered_results.append(new_row if len(new_row) > 1 else row)

                logger.info(
                    f"Applied Python-side filtering for {target_keywords} "
                    "without dropping compact cross-omics rows."
                )
                results = filtered_results
            if results:
                for row in results:
                    for k, v in row.items():
                        if isinstance(v, decimal.Decimal):
                            row[k] = float(v)
            
            is_gene_density = "gene density" in user_question_text or "distribution" in user_question_text
            if is_gene_density and "start_bp" in sql_text.lower() and len(results) > 100:
                result_str = f"Successfully retrieved {len(results)} gene coordinates for visualization. [DATA HIDDEN TO SAVE TOKENS]. PLEASE REPLY TO USER: 'I have generated the gene density heatmap for the top chromosomes as requested.'"
                logger.info(f"Triggered bypass for massive coordinate data. Hidden {len(results)} rows from LLM context.")
            else:
                result_str = json.dumps(results, ensure_ascii=False, indent=2)
                
                if len(result_str) > 12000:
                    result_str = result_str[:12000] + "... [truncated]"

            return {
                "messages": [
                    AIMessage(
                        content=f"[SQL]: {sql_query}\n[Result]:\n{result_str}",
                        additional_kwargs={
                            "type": "salvia_db_response",
                            "results": results,
                            "result_batches": result_batches,
                        }
                    )
                ],
                "metadata": {
                    **metadata, 
                    "thinking_content": f"Database query executed successfully. Found {len(results)} records."
                },
                "extracted_data": results  # <--- 画图引擎 (Generator) 将使用这个未被截断的完整列表！
            }
            
        except Exception as e:
            logger.error(f"Database Execution Error: {e}")
            return {
                "messages": [
                    AIMessage(
                        content=f"Database error: {str(e)}", 
                        additional_kwargs={"type": "salvia_error"}
                    )
                ]
            }
