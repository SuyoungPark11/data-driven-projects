# dacon-hansol-deco-3
## 프로젝트 설명

```
📂 configs : run.py을 작동하기 위한 config 파일. 각자 환경에 맞게끔 자신의 yaml 생성해서 사용.
📂 utils : run.py
    - 📜 data_utils.py : 데이터 처리를 위한 함수 모음. 데이터 로드, 전처리, qa data 생성 등등
    - 📜 model_utils.py : 모델 관련 처리를 위한 함수 모음. 모델 로드, 벡터 스토어 생성, RAG Chain 생성 등
📜 run.py : 베이스라인 작동을 위한 함수 아래 실행 방법 참고. utils에 있는 함수들을 가져와서 추론 및 제출 파일 생성.
```

## 실행 방법

config 파일 경로의 경우 자신의 환경에 맞춰서 yaml 파일 생성후 수정

```bash
python run.py --cfg-path ./configs/example.yaml

```
