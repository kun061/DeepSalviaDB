# 文件名: structure_service.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests

# 1. 初始化当前模块的路由器
router = APIRouter()

# 2. 定义前端传来的数据模型
class SequenceRequest(BaseModel):
    sequence: str

# 3. 编写预测接口逻辑
# 注意：这里的路径是 "/predict_structure"
# 等会儿在 main.py 里挂载了 prefix 之后，完整路径就会变成 "/structure/predict_structure"
@router.post("/predict_structure")
async def predict_structure(req: SequenceRequest):
    sequence = req.sequence.strip()
    if not sequence:
        raise HTTPException(status_code=400, detail="蛋白质序列不能为空")

    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    print(f"[Structure Service] 正在向 ESM 提交序列进行 3D 预测 (长度: {len(sequence)})...")
    
    try:
        response = requests.post(
            url, 
            data=sequence, 
            headers={"Content-Type": "text/plain"},
            timeout=60 # ESM 计算可能需要几十秒
        )
        
        if response.status_code == 200:
            print("[Structure Service] PDB 数据获取成功！")
            return {"status": "success", "pdb_data": response.text}
        else:
            raise HTTPException(status_code=response.status_code, detail=f"预测失败: {response.text}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))