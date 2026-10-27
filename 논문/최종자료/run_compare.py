"""비교 실험 노트북(비교실험_선행연구_평가설계.ipynb, E1~E3)을 새 커널에서 위에서부터 순서대로 전체 실행하고 출력을 저장합니다.

사용법:  python run_compare.py   (먼저 python run_final.py)
- 이 파일, 두 노트북, 데이터/ 폴더가 같은 위치에 있어야 합니다.
- 주 분석 노트북의 1~9장을 다시 실행하고, 주 분석이 저장한 최종 설정(outputs_final/summary.json)을 읽습니다.
- 결과는 outputs_compare/, figures_compare/에 저장하며 주 분석 결과 파일은 바꾸지 않습니다.
- 실행 전에 이전 출력을 모두 지웁니다. 오류가 나면 0이 아닌 코드로 끝납니다.
"""
import os
import sys
import time
import asyncio

import nbformat
from nbclient import NotebookClient

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ["PYTHONNOUSERSITE"] = "1"   # 사용자 폴더(AppData) 패키지를 섞지 않음: 커널·병렬 작업 모두 anaconda 패키지만 사용

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "비교실험_선행연구_평가설계.ipynb")
if not os.path.exists(os.path.join(HERE, "outputs_final", "summary.json")):
    print("outputs_final/summary.json이 없습니다. 먼저 python run_final.py를 실행하세요.")
    sys.exit(1)

nb = nbformat.read(NB, as_version=4)
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs, c.execution_count = [], None
t0 = time.time()
client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": HERE}})
try:
    client.execute()
    print(f"OK: 전체 실행 완료 ({time.time() - t0:.0f}초)")
except Exception as e:
    print("실패:", type(e).__name__, str(e)[-3000:])
    sys.exit(1)
finally:
    nbformat.write(nb, NB)
