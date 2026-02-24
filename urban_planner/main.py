"""
한국 도시설계 토지이용계획 수립 툴
FastAPI 메인 애플리케이션
실행: uvicorn main:app --reload --port 8000
"""

import os
import traceback
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Change working directory to the app directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from engine import (
    ContextFetcher,
    SiteAnalyzer,
    MasterPlanner,
    RoadGenerator,
    BlockDivider,
    FacilityPlacer,
    PlanValidator,
)

app = FastAPI(
    title="한국 도시설계 토지이용계획 수립 툴",
    description="Korean Urban Planning Land Use Planning Tool",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class PlanRequest(BaseModel):
    boundary_geojson: dict
    site_name: str = "테스트 구역"
    development_type: str = "신규개발"
    program: str = "혼합용도"
    target_population: int = 0
    target_far: float = 0
    green_ratio_min: float = 0.10
    road_ratio_target: float = 0.25
    region: str = "서울"
    transit_access: str = "없음"
    priority: dict = {
        "density": 3,
        "green": 3,
        "walkability": 3,
        "publicfacility": 3,
        "traffic": 3,
    }


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/analyze")
async def analyze(request: PlanRequest):
    try:
        fetcher = ContextFetcher()
        context = fetcher.fetch_context(request.boundary_geojson)
        analyzer = SiteAnalyzer()
        result = analyzer.analyze(request.boundary_geojson, context)
        return {"success": True, "data": result}
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.post("/api/plan")
async def create_plan(request: PlanRequest):
    try:
        # 1. 컨텍스트 수집
        fetcher = ContextFetcher()
        context_raw = fetcher.fetch_context(
            request.boundary_geojson, buffer_m=500
        )
        entry_points = fetcher.get_entry_points(
            request.boundary_geojson,
            context_raw['roads_gdf']
        )
        context_analysis = fetcher.analyze_surroundings(
            request.boundary_geojson,
            context_raw['roads_gdf'],
            context_raw['landuse_gdf'],
            context_raw['facilities_gdf'],
            context_raw['subway_stations']
        )

        # 2. 현황 분석
        analyzer = SiteAnalyzer()
        site_analysis = analyzer.analyze(
            request.boundary_geojson,
            {
                'context': context_analysis,
                'entry_points': entry_points,
                'development_type': request.development_type,
                'program': request.program,
            }
        )

        # 3. 도로망 생성
        road_gen = RoadGenerator()
        road_result = road_gen.generate(
            request.boundary_geojson,
            entry_points,
            target_road_ratio=request.road_ratio_target
        )

        # 4. 블록 분할
        divider = BlockDivider()
        block_result = divider.divide(
            request.boundary_geojson,
            road_result['road_polygons_gdf']
        )

        # 5. 토지이용계획
        planner = MasterPlanner()
        design_params = request.dict()
        # Pass blocks to planner for zoning assignment
        design_params['_blocks_gdf'] = block_result['blocks_gdf']
        plan_result = planner.plan(
            request.boundary_geojson,
            site_analysis,
            design_params,
            context_analysis
        )

        # 6. 공공시설 배치
        placer = FacilityPlacer()
        facility_result = placer.place(
            block_result['blocks_gdf'],
            design_params,
            context_raw
        )

        # 7. 법규 검증
        validator = PlanValidator()
        validation = validator.validate(
            request.boundary_geojson,
            road_result['road_stats'],
            block_result['blocks_gdf'],
            facility_result['facilities_gdf'],
            plan_result['zoning_stats']
        )

        # Build context GeoJSON (safely)
        context_geojson = {}
        try:
            if not context_raw['roads_gdf'].empty:
                context_geojson['roads'] = context_raw['roads_gdf'].to_crs(4326).to_json()
            else:
                context_geojson['roads'] = None
        except Exception:
            context_geojson['roads'] = None

        try:
            if not context_raw['landuse_gdf'].empty:
                context_geojson['landuse'] = context_raw['landuse_gdf'].to_crs(4326).to_json()
            else:
                context_geojson['landuse'] = None
        except Exception:
            context_geojson['landuse'] = None

        return {
            "success": True,
            "data": {
                "site_analysis": site_analysis,
                "context_analysis": context_analysis,
                "entry_points": entry_points,
                "context_geojson": context_geojson,
                "roads": road_result['road_geojson'],
                "blocks": block_result['blocks_geojson'],
                "zoning": plan_result['zoning_geojson'],
                "facilities": facility_result['facilities_geojson'],
                "validation": validation,
                "road_stats": road_result['road_stats'],
                "block_stats": block_result['block_stats'],
                "zoning_stats": plan_result['zoning_stats'],
                "coverage": facility_result['coverage'],
                "design_rationale": plan_result['design_rationale'],
                "subway_stations": context_raw['subway_stations'],
                "bus_stops": context_raw['bus_stops'],
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "Urban Planning Tool is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
