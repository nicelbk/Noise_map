"""
한국 도시설계 토지이용계획 수립 툴
FastAPI 서버 진입점

실행: uvicorn main:app --reload --port 8000
접속: http://localhost:8000
"""

import json
import io
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import (
    HTMLResponse, FileResponse, StreamingResponse, JSONResponse
)
from pydantic import BaseModel

from engine.site_analyzer import SiteAnalyzer
from engine.master_planner import MasterPlanner
from engine.road_generator import RoadGenerator
from engine.block_divider import BlockDivider
from engine.facility_placer import FacilityPlacer
from engine.validator import PlanValidator
from engine.visualizer import Visualizer

app = FastAPI(
    title="한국 도시설계 토지이용계획 수립 툴",
    description="GIS 기반 자동 토지이용계획 수립 시스템",
    version="1.0.0",
)

# static 파일 마운트
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# output 디렉터리
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# 엔진 인스턴스
site_analyzer = SiteAnalyzer()
master_planner = MasterPlanner()
road_generator = RoadGenerator()
block_divider = BlockDivider()
facility_placer = FacilityPlacer()
validator = PlanValidator()
visualizer = Visualizer()

# 마지막 계획 결과 저장 (내보내기용)
_last_plan: dict = {}


# ─────────────────────────────────────────
# 요청/응답 모델
# ─────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    boundary_geojson: dict
    existing_data: Optional[dict] = None
    design_params: Optional[dict] = None


class PlanRequest(BaseModel):
    boundary_geojson: dict
    design_params: dict


class ValidateRequest(BaseModel):
    plan_geojson: dict


class NoiseRequest(BaseModel):
    plan_geojson: dict


class AirRequest(BaseModel):
    plan_geojson: dict


# ─────────────────────────────────────────
# 루트 페이지
# ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지 (index.html)"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html을 찾을 수 없습니다.</h1>")


# ─────────────────────────────────────────
# API 엔드포인트
# ─────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_site(request: AnalyzeRequest):
    """
    Stage 0: 대상지 현황 분석

    - 면적, 형상지수 계산
    - 유사 사례 매칭
    - 제약조건 분석
    """
    try:
        # 면적 사전 검사
        boundary = request.boundary_geojson
        if not _is_valid_geojson(boundary):
            raise HTTPException(status_code=400, detail="유효하지 않은 GeoJSON입니다.")

        result = site_analyzer.analyze(
            boundary_geojson=boundary,
            design_params=request.design_params or {},
        )

        # 오류 조건 확인
        for constraint in result.get("constraints", []):
            if constraint.get("level") == "error" and "TOO_SMALL" in constraint.get("code", ""):
                raise HTTPException(status_code=400, detail=constraint["message"])

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")


@app.post("/api/plan")
async def create_plan(request: PlanRequest):
    """
    전체 토지이용계획 수립

    Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → 검증
    """
    global _last_plan

    try:
        boundary = request.boundary_geojson
        params = request.design_params

        if not _is_valid_geojson(boundary):
            raise HTTPException(status_code=400, detail="유효하지 않은 GeoJSON입니다.")

        # ── Stage 0: 현황 분석 ──────────────────
        site_analysis = site_analyzer.analyze(
            boundary_geojson=boundary,
            design_params=params,
        )

        # 면적 검사
        area_m2 = site_analysis.get("area_m2", 0)
        if area_m2 < 1000:
            raise HTTPException(
                status_code=400,
                detail=f"대상지 면적({area_m2:.0f}m²)이 너무 작습니다. 최소 1,000m² 이상이어야 합니다."
            )
        if area_m2 > 10_000_000:  # 1,000ha
            raise HTTPException(
                status_code=400,
                detail="대상지 면적이 너무 큽니다. 100ha 이하를 권장합니다."
            )

        # ── Stage 1: 마스터플랜 ─────────────────
        master_plan = master_planner.plan(
            boundary_geojson=boundary,
            site_analysis=site_analysis,
            design_params=params,
        )

        # ── Stage 2: 도로망 생성 ────────────────
        road_result = road_generator.generate(
            boundary_geojson=boundary,
            master_plan=master_plan,
            design_params=params,
        )

        # ── Stage 3: 블록 분할 ──────────────────
        blocks = block_divider.divide(
            boundary_geojson=boundary,
            road_polygons=road_result["road_polygons"],
            master_plan=master_plan,
            design_params=params,
        )

        # ── Stage 4: 공공시설 배치 ──────────────
        facilities_result = facility_placer.place(
            blocks_geojson=blocks,
            design_params=params,
            site_analysis=site_analysis,
        )

        # ── 통계 계산 ───────────────────────────
        statistics = _calc_statistics(
            blocks, road_result["road_stats"],
            facilities_result, site_analysis
        )

        # ── 법규 검증 ───────────────────────────
        full_plan_for_validation = {
            "master_plan": master_plan,
            "road_network": road_result["road_network"],
            "road_polygons": road_result["road_polygons"],
            "blocks": blocks,
            "facilities": facilities_result,
            "road_stats": road_result["road_stats"],
            "site_analysis": site_analysis,
            "statistics": statistics,
        }
        validation = validator.validate(full_plan_for_validation)

        # ── 대안 생성 ───────────────────────────
        alternatives = master_planner.generate_alternatives(
            boundary_geojson=boundary,
            site_analysis=site_analysis,
            base_params=params,
        )

        # 통계 리포트 텍스트
        full_result = {
            **full_plan_for_validation,
            "boundary_geojson": boundary,
            "site_name": params.get("site_name", "대상지"),
            "validation": validation,
            "alternatives": {
                k: v["plan"] for k, v in alternatives.items()
            },
            "alternative_labels": {k: v["label"] for k, v in alternatives.items()},
            "similar_cases": site_analysis.get("similar_cases", []),
        }

        report_text = visualizer.generate_statistics_report(full_result)

        # 마지막 계획 저장 (내보내기용)
        _last_plan = full_result

        # 저장
        _save_plan(full_result, params.get("site_name", "plan"))

        return JSONResponse(content={
            "site_analysis": site_analysis,
            "master_plan": master_plan,
            "road_network": road_result["road_network"],
            "road_polygons": road_result["road_polygons"],
            "road_stats": road_result["road_stats"],
            "blocks": blocks,
            "facilities": facilities_result["facilities"],
            "coverage_analysis": facilities_result["coverage_analysis"],
            "estimates": facilities_result["estimates"],
            "validation": validation,
            "statistics": statistics,
            "similar_cases": site_analysis.get("similar_cases", []),
            "alternatives": {k: v["plan"] for k, v in alternatives.items()},
            "alternative_labels": {k: v["label"] for k, v in alternatives.items()},
            "report_text": report_text,
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"계획 수립 중 오류: {str(e)}\n{tb}")


@app.post("/api/validate")
async def validate_plan(request: ValidateRequest):
    """계획 법규 검증"""
    try:
        result = validator.validate(request.plan_geojson)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/{format}")
async def export_plan(format: str):
    """
    계획 결과 내보내기

    format: geojson | png
    """
    global _last_plan

    if not _last_plan:
        raise HTTPException(status_code=404, detail="내보낼 계획이 없습니다. 먼저 계획을 수립해주세요.")

    if format == "geojson":
        # GeoJSON 형태로 모든 레이어 묶어서 반환
        export_data = {
            "type": "FeatureCollection",
            "metadata": {
                "site_name": _last_plan.get("site_name", ""),
                "generated_at": _get_timestamp(),
            },
            "layers": {
                "master_plan": _last_plan.get("master_plan", {}),
                "roads": _last_plan.get("road_polygons", {}),
                "blocks": _last_plan.get("blocks", {}),
                "facilities": _last_plan.get("facilities", {}),
            }
        }
        content = json.dumps(export_data, ensure_ascii=False, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/geo+json",
            headers={"Content-Disposition": "attachment; filename=urban_plan.geojson"}
        )

    elif format == "png":
        try:
            img_base64 = visualizer.render_plan(_last_plan)
            img_bytes = base64.b64decode(img_base64)
            return StreamingResponse(
                io.BytesIO(img_bytes),
                media_type="image/png",
                headers={"Content-Disposition": "attachment; filename=urban_plan.png"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"이미지 생성 오류: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 형식: {format}. (geojson, png 지원)")


# ─────────────────────────────────────────
# AI 모델 연동 인터페이스 (stub)
# 나중에 소음/대기 예측 AI와 연동 예정
# ─────────────────────────────────────────

@app.post("/api/predict/noise")
async def predict_noise(request: NoiseRequest):
    """
    소음 예측 (AI 모델 연동 인터페이스 - 현재 stub)

    실제 구현 시 소음 예측 AI 모델과 연동:
    - 도로 교통 소음: 교통량, 차종, 도로 선형
    - 철도 소음
    - 항공기 소음
    - 공장 소음
    """
    return JSONResponse(content={
        "status": "not_implemented",
        "message": "소음 예측 AI 모델 연동 예정 (POST /api/predict/noise)",
        "interface": {
            "input": {
                "plan_geojson": "FeatureCollection - 도로, 건물, 용도지역 레이어",
                "traffic_volume": "dict - 시간대별 교통량",
                "road_types": "list - 도로 유형별 교통 특성",
            },
            "output": {
                "noise_map": "base64 PNG - 소음 등고선 지도",
                "stats": {
                    "max_db": "float - 최대 소음도 (dB)",
                    "avg_db": "float - 평균 소음도 (dB)",
                    "exceedance_ratio": "float - 기준 초과 면적 비율",
                    "affected_population": "int - 영향 인구 수",
                }
            }
        },
        "noise_map": None,
        "stats": {
            "max_db": None,
            "avg_db": None,
            "exceedance_ratio": None,
            "affected_population": None,
        }
    })


@app.post("/api/predict/air")
async def predict_air(request: AirRequest):
    """
    대기질 예측 (AI 모델 연동 인터페이스 - 현재 stub)

    실제 구현 시 대기질 예측 AI 모델과 연동:
    - PM2.5, PM10 농도 분포
    - NOx, SOx 분포
    - 바람장 기반 확산 예측
    """
    return JSONResponse(content={
        "status": "not_implemented",
        "message": "대기질 예측 AI 모델 연동 예정 (POST /api/predict/air)",
        "interface": {
            "input": {
                "plan_geojson": "FeatureCollection - 도로, 건물, 녹지 레이어",
                "emission_sources": "list - 오염원 위치 및 배출량",
                "meteorological_data": "dict - 기상 데이터 (풍향, 풍속, 안정도)",
            },
            "output": {
                "air_map": "base64 PNG - 대기질 농도 분포 지도",
                "stats": {
                    "pm25_avg": "float - PM2.5 평균 농도 (μg/m³)",
                    "pm10_avg": "float - PM10 평균 농도 (μg/m³)",
                    "exceedance_ratio": "float - 환경기준 초과 면적 비율",
                }
            }
        },
        "air_map": None,
        "stats": {
            "pm25_avg": None,
            "pm10_avg": None,
            "exceedance_ratio": None,
        }
    })


# ─────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────

def _is_valid_geojson(geojson: dict) -> bool:
    """GeoJSON 유효성 검사"""
    if not isinstance(geojson, dict):
        return False
    geo_type = geojson.get("type")
    if geo_type == "FeatureCollection":
        return bool(geojson.get("features"))
    if geo_type == "Feature":
        return bool(geojson.get("geometry"))
    if geo_type in ("Polygon", "MultiPolygon"):
        return bool(geojson.get("coordinates"))
    return False


def _calc_statistics(blocks, road_stats, facilities_result, site_analysis) -> dict:
    """전체 계획 통계 계산"""
    total_area = site_analysis.get("area_m2", 1)

    # 블록별 면적 집계
    area_by_category = {}
    for f in blocks.get("features", []):
        cat = f["properties"].get("category", "기타")
        area = f["properties"].get("area_m2", 0)
        area_by_category[cat] = area_by_category.get(cat, 0) + area

    # 도로 면적 추가
    area_by_category["도로"] = road_stats.get("road_area_m2", 0)

    estimates = facilities_result.get("estimates", {})

    return {
        "total_area_m2": round(total_area, 1),
        "total_area_ha": round(total_area / 10000, 3),
        "area_by_category": {
            k: {
                "area_m2": round(v, 1),
                "ratio_pct": round(v / total_area * 100, 1) if total_area > 0 else 0,
            }
            for k, v in area_by_category.items()
        },
        "road": {
            "total_length_m": road_stats.get("total_length_m", 0),
            "area_m2": road_stats.get("road_area_m2", 0),
            "ratio_pct": round(road_stats.get("road_ratio", 0) * 100, 1),
        },
        "population": {
            "households": estimates.get("households", 0),
            "population": estimates.get("population", 0),
        },
        "facilities_count": {
            f["properties"].get("facility_type", "기타"): (
                facilities_result.get("facilities", {}).get("features", []).count(f)
            )
            for f in facilities_result.get("facilities", {}).get("features", [])
        },
    }


def _save_plan(plan: dict, site_name: str):
    """계획 결과를 output/ 디렉터리에 저장"""
    try:
        save_path = output_dir / f"{site_name}_plan.json"
        with open(save_path, "w", encoding="utf-8") as fp:
            # GeoJSON 지오메트리는 직렬화 가능하므로 그대로 저장
            json.dump(plan, fp, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass  # 저장 실패 시 무시


def _get_timestamp() -> str:
    """현재 시각 문자열 반환"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
