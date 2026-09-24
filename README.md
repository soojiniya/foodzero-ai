# FoodZero

공공데이터 기반 음식물쓰레기 발생량 예측 및 스마트 수거·감축 지원 플랫폼

## Project Link


🌐 [Live Demo](https://foodzero-ai.streamlit.app/)

# FoodZero AI 데이터 파이프라인

지자체별 일별 음식물쓰레기 배출량을 기상, 인구, 날짜 특성으로 예측하기 위한 데이터 수집 및 전처리 단계입니다. 이번 범위는 원본 검사, 기상 데이터 수집, 지자체-ASOS 관측소 매핑, 인구 결합, 파생변수 생성, 최종 학습 데이터셋 생성까지입니다.

## 원본 데이터 배치

- 음식물쓰레기: `data/raw/food_waste`
- 주민등록 인구 및 세대현황: `data/raw/population`
- ASOS 관측소 매핑: `config/asos_station_mapping.csv`

현재 workspace에는 원본 CSV/Excel이 없어서 `docs/data_audit.md`는 파일 미존재 상태를 기록합니다. 원본 파일을 위 폴더에 넣은 뒤 감사 명령을 다시 실행하면 실제 컬럼명, 인코딩, 행 수, 결측치, 샘플 지자체명이 문서화됩니다.

## 환경 변수

`.env` 파일에 공공데이터포털 인증키를 저장합니다. 키는 코드에 직접 쓰지 않습니다.

```text
KMA_API_KEY=...
```

`.env`는 `.gitignore`에 포함되어 있고, 예시는 `.env.example`에 있습니다.

## 실행

프로젝트 공식 실행 환경은 `.venv`입니다.

```powershell
.\.venv\Scripts\python.exe -m pipelines.run_pipeline
```

이미 기상 캐시가 있으면:

```powershell
.\.venv\Scripts\python.exe -m pipelines.run_pipeline --skip-weather
```

## 기상 데이터

기상청 `지상(종관, ASOS) 일자료 조회서비스`를 사용합니다.

- 기본 URL: `https://apis.data.go.kr/1360000/AsosDalyInfoService`
- endpoint: `/getWthrDataList`
- 고정 파라미터: `dataCd=ASOS`, `dateCd=DAY`, `dataType=JSON`
- 기간: `2021-01-01`부터 `2024-01-31`
- 우선 변수: 평균기온 `avgTa`, 최고기온 `maxTa`, 최저기온 `minTa`, 일강수량 `sumRn`, 평균상대습도 `avgRhm`

API 응답은 `data/interim/weather/asos_daily_{stn_id}_{start}_{end}.json`에 캐시합니다. 통합 CSV는 `data/interim/weather/asos_daily_weather.csv`입니다. API 오류는 `data/interim/weather/weather_errors.json`에 저장합니다.

## 지자체-ASOS 관측소 매핑

`config/asos_station_mapping.csv`는 별도 관리합니다. 임의 배정을 피하기 위해 `mapping_status=confirmed`인 행만 최종 데이터셋에 사용합니다.

필수 기록:

- `sigungu_key`: 원본의 시도와 시군구를 결합한 키
- `stn_id`, `stn_name`: ASOS 관측소 ID와 이름
- `mapping_status`: `confirmed`, `needs_review`, `not_applicable`
- `evidence`: 근거. 예: 기상자료개방포털 관측소 위치, 지자체 중심지와 관측소 거리, 행정구역 경계 검토 결과
- `source_url`: 근거 URL
- `notes`: ASOS가 적절하지 않은 지역 또는 AWS/인근 관측소 검토 필요 사유

`needs_review` 또는 `not_applicable` 지역은 최종 학습 데이터셋 생성 시 사용하지 않습니다.

원본 음식물쓰레기 파일에서 실제 지자체 목록을 읽어 매핑 입력 템플릿을 만들 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m src.foodzero.mapping
```

출력: `config/asos_station_mapping_template_from_raw.csv`

## 예측 목표 설계

최종 데이터셋의 기본 목표값은 `waste_amount`, 즉 해당일 배출량입니다. 이유는 이번 기상 소스가 ASOS 일자료의 실제 관측값이므로 해당일의 실제 기상값과 해당일 배출량을 결합하는 회귀 데이터셋에 적합하기 때문입니다. 다음날 사전 예측을 운영 목표로 삼으려면 ASOS 관측값 대신 예보 기상 데이터나 예측 시점에 이미 확정된 전날 기상값만 사용하도록 별도 설계가 필요합니다.

데이터 누수를 막기 위해 `lag_1`, `rolling_mean_7`, `rolling_mean_14`는 모두 지자체별 날짜 정렬 후 `shift(1)`을 먼저 적용한 과거 배출량으로만 계산합니다.

## 산출물

- 원본 감사 문서: `docs/data_audit.md`
- 원본 감사 JSON: `data/interim/source_inventory.json`
- 기상 캐시: `data/interim/weather`
- 최종 학습 데이터: `data/processed/foodzero_model_dataset.csv`

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## FoodZero AI 웹서비스 실행

Streamlit 기반 대시보드는 `src/web_app.py`에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/web_app.py
```

새 환경을 구성해야 하는 경우 다음 의존성을 설치한 뒤 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run src/web_app.py
```

웹앱은 다음 산출물을 사용합니다.

- 모델: `models/foodzero_next_day_model.joblib`
- 서비스 feature: `data/processed/next_day_service_features.csv`
- 평가 결과: `evaluation/next_day/`

기존 연구용 historical/backtest 모델 `models/foodzero_final_model.joblib`과 기존 `evaluation/` 결과는 보존합니다.
