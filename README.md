# Combat_Balance_GUI
# Combat Balance Analyzer

전투 밸런스를 분석하고 최적화하는 GUI 기반 도구입니다.

## 기능

- PC1과 PC2의 전투 데이터 로드 및 분석
- 전투 시뮬레이션 및 결과 예측
- 신경망 기반 전투 시간 및 승자 예측
- 전투 밸런스 통계 분석
- 캐릭터 스탯 최적화

## 설치 방법

1. Python 3.8 이상이 설치되어 있어야 합니다.
2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 프로그램 실행:
```bash
python Combat_Balance_GUI.py
```

2. 데이터 로드:
   - PC1 CSV 파일 선택: HP1, Attack1, Defense1 열이 포함된 CSV 파일
   - PC2 CSV 파일 선택: HP2, Attack2, Defense2 열이 포함된 CSV 파일

3. 학습 설정:
   - 에포크 수 설정 (기본값: 100)
   - 최적화 단계 설정 (기본값: 100)

4. 모델 학습 시작:
   - "모델 학습 시작" 버튼 클릭
   - 학습 진행 상황을 실시간으로 확인 가능

5. 결과 분석:
   - 전투 밸런스 통계 확인
   - 그래프를 통한 시각적 분석
   - 캐릭터 스탯 최적화 수행

## CSV 파일 형식

### PC1 데이터 (예시)
```csv
HP1,Attack1,Defense1
1000,50,20
950,45,18
1050,55,22
```

### PC2 데이터 (예시)
```csv
HP2,Attack2,Defense2
900,45,18
850,40,16
950,50,20
```

## 주의사항

- CSV 파일은 반드시 지정된 열 이름을 포함해야 합니다.
- 데이터는 숫자 형식이어야 합니다.
- PC1과 PC2 데이터의 행 수가 일치해야 합니다. 
