"""
대상지 주변 500m OSM 데이터 수집
모든 메서드는 아래 시그니처를 정확히 따른다
"""

import math
import json
import requests
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Point, LineString, Polygon, MultiLineString
from shapely.ops import unary_union
from pyproj import Transformer


EPSG_WGS84 = 4326
EPSG_5179 = 5179

# WGS84 → EPSG:5179
transformer_to_5179 = Transformer.from_crs(EPSG_WGS84, EPSG_5179, always_xy=True)
# EPSG:5179 → WGS84
transformer_to_wgs84 = Transformer.from_crs(EPSG_5179, EPSG_WGS84, always_xy=True)


def _empty_gdf(crs=EPSG_5179) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{crs}")


class ContextFetcher:

    def fetch_context(
        self,
        boundary_geojson: dict,
        buffer_m: float = 500
    ) -> dict:
        """
        반환:
        {
          'roads_gdf': GeoDataFrame(crs=EPSG:5179),
          'landuse_gdf': GeoDataFrame(crs=EPSG:5179),
          'facilities_gdf': GeoDataFrame(crs=EPSG:5179),
          'subway_stations': list[dict],
          'bus_stops': list[dict],
          'entry_points': list[dict],
          'context_summary': dict
        }
        실패시: 빈 GeoDataFrame과 빈 리스트 반환
        절대 예외 발생시키지 않음
        """
        try:
            bbox = self._boundary_to_bbox(boundary_geojson, buffer_m)
            osm_data = self._fetch_osm(bbox)

            roads_gdf = self._parse_roads(osm_data)
            landuse_gdf = self._parse_landuse(osm_data)
            facilities_gdf = self._parse_facilities(osm_data)
            subway_stations = self._parse_subway_stations(osm_data)
            bus_stops = self._parse_bus_stops(osm_data)
            entry_points = self.get_entry_points(boundary_geojson, roads_gdf)

            context_summary = {
                'road_count': len(roads_gdf),
                'landuse_count': len(landuse_gdf),
                'facility_count': len(facilities_gdf),
                'subway_count': len(subway_stations),
                'bus_stop_count': len(bus_stops),
                'entry_point_count': len(entry_points),
                'bbox': bbox,
            }

            return {
                'roads_gdf': roads_gdf,
                'landuse_gdf': landuse_gdf,
                'facilities_gdf': facilities_gdf,
                'subway_stations': subway_stations,
                'bus_stops': bus_stops,
                'entry_points': entry_points,
                'context_summary': context_summary,
            }
        except Exception:
            return {
                'roads_gdf': _empty_gdf(),
                'landuse_gdf': _empty_gdf(),
                'facilities_gdf': _empty_gdf(),
                'subway_stations': [],
                'bus_stops': [],
                'entry_points': [],
                'context_summary': {},
            }

    def _boundary_to_bbox(
        self,
        boundary_geojson: dict,
        buffer_m: float
    ) -> tuple:
        """
        반환: (south, west, north, east) WGS84
        """
        geom = shape(boundary_geojson)
        # Convert to EPSG:5179 for buffer in meters
        coords = list(geom.exterior.coords)
        transformed = [transformer_to_5179.transform(lon, lat) for lon, lat in coords]
        poly_5179 = Polygon(transformed)
        buffered_5179 = poly_5179.buffer(buffer_m)
        # Convert back to WGS84
        buf_coords = list(buffered_5179.exterior.coords)
        wgs84_coords = [transformer_to_wgs84.transform(x, y) for x, y in buf_coords]
        lons = [c[0] for c in wgs84_coords]
        lats = [c[1] for c in wgs84_coords]
        return (min(lats), min(lons), max(lats), max(lons))

    def _fetch_osm(
        self,
        bbox: tuple,
        timeout: int = 25
    ) -> dict:
        """
        Overpass API 호출
        실패/타임아웃시 빈 dict 반환
        """
        try:
            south, west, north, east = bbox
            overpass_url = "https://overpass-api.de/api/interpreter"
            query = f"""
[out:json][timeout:{timeout}];
(
  way["highway"]({south},{west},{north},{east});
  way["landuse"]({south},{west},{north},{east});
  way["building"]({south},{west},{north},{east});
  node["railway"="station"]({south},{west},{north},{east});
  node["railway"="subway_entrance"]({south},{west},{north},{east});
  node["subway"="yes"]({south},{west},{north},{east});
  node["highway"="bus_stop"]({south},{west},{north},{east});
  way["leisure"="park"]({south},{west},{north},{east});
  way["amenity"]({south},{west},{north},{east});
  node["amenity"]({south},{west},{north},{east});
);
out body;
>;
out skel qt;
"""
            response = requests.post(
                overpass_url,
                data={'data': query},
                timeout=timeout
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}

    def _parse_roads(self, osm_data: dict) -> gpd.GeoDataFrame:
        """OSM 데이터에서 도로 추출"""
        try:
            if not osm_data or 'elements' not in osm_data:
                return _empty_gdf()

            nodes = {e['id']: e for e in osm_data['elements'] if e['type'] == 'node'}
            roads = []

            highway_width_map = {
                'motorway': 30, 'trunk': 25, 'primary': 20,
                'secondary': 15, 'tertiary': 12, 'residential': 8,
                'service': 6, 'unclassified': 8, 'living_street': 6,
                'footway': 3, 'cycleway': 3, 'path': 3,
            }

            for elem in osm_data['elements']:
                if elem['type'] != 'way':
                    continue
                tags = elem.get('tags', {})
                highway = tags.get('highway')
                if not highway:
                    continue

                node_ids = elem.get('nodes', [])
                coords = []
                for nid in node_ids:
                    if nid in nodes:
                        n = nodes[nid]
                        coords.append((n['lon'], n['lat']))

                if len(coords) < 2:
                    continue

                x_coords = [transformer_to_5179.transform(lon, lat)[0] for lon, lat in coords]
                y_coords = [transformer_to_5179.transform(lon, lat)[1] for lon, lat in coords]
                line = LineString(zip(x_coords, y_coords))

                width_m = highway_width_map.get(highway, 8)
                try:
                    width_tag = float(tags.get('width', 0))
                    if width_tag > 0:
                        width_m = width_tag
                except (ValueError, TypeError):
                    pass

                roads.append({
                    'geometry': line,
                    'highway': highway,
                    'name': tags.get('name', tags.get('name:ko', '')),
                    'width_m': width_m,
                    'osm_id': elem['id'],
                })

            if not roads:
                return _empty_gdf()

            gdf = gpd.GeoDataFrame(roads, crs=f"EPSG:{EPSG_5179}")
            return gdf
        except Exception:
            return _empty_gdf()

    def _parse_landuse(self, osm_data: dict) -> gpd.GeoDataFrame:
        """OSM 데이터에서 토지이용 추출"""
        try:
            if not osm_data or 'elements' not in osm_data:
                return _empty_gdf()

            nodes = {e['id']: e for e in osm_data['elements'] if e['type'] == 'node'}
            landuses = []

            for elem in osm_data['elements']:
                if elem['type'] != 'way':
                    continue
                tags = elem.get('tags', {})
                landuse = tags.get('landuse') or tags.get('leisure') or tags.get('amenity')
                if not landuse:
                    continue

                node_ids = elem.get('nodes', [])
                coords = []
                for nid in node_ids:
                    if nid in nodes:
                        n = nodes[nid]
                        coords.append((n['lon'], n['lat']))

                if len(coords) < 3:
                    continue

                x_coords = [transformer_to_5179.transform(lon, lat)[0] for lon, lat in coords]
                y_coords = [transformer_to_5179.transform(lon, lat)[1] for lon, lat in coords]
                poly = Polygon(zip(x_coords, y_coords))
                if not poly.is_valid:
                    poly = poly.buffer(0)

                landuses.append({
                    'geometry': poly,
                    'landuse': landuse,
                    'name': tags.get('name', tags.get('name:ko', '')),
                    'osm_id': elem['id'],
                })

            if not landuses:
                return _empty_gdf()

            gdf = gpd.GeoDataFrame(landuses, crs=f"EPSG:{EPSG_5179}")
            return gdf
        except Exception:
            return _empty_gdf()

    def _parse_facilities(self, osm_data: dict) -> gpd.GeoDataFrame:
        """OSM 데이터에서 시설 추출"""
        try:
            if not osm_data or 'elements' not in osm_data:
                return _empty_gdf()

            nodes = {e['id']: e for e in osm_data['elements'] if e['type'] == 'node'}
            facilities = []

            facility_tags = ['school', 'hospital', 'library', 'community_centre',
                             'police', 'fire_station', 'post_office', 'bank']

            for elem in osm_data['elements']:
                tags = elem.get('tags', {})
                amenity = tags.get('amenity', '')

                if elem['type'] == 'node' and amenity in facility_tags:
                    lon, lat = elem.get('lon', 0), elem.get('lat', 0)
                    x, y = transformer_to_5179.transform(lon, lat)
                    # Create small square polygon (50m x 50m) for node facilities
                    pt = Point(x, y)
                    poly = pt.buffer(25, cap_style=3)  # square buffer
                    facilities.append({
                        'geometry': poly,
                        'facility_type': amenity,
                        'name': tags.get('name', tags.get('name:ko', '')),
                    })
                elif elem['type'] == 'way' and amenity in facility_tags:
                    node_ids = elem.get('nodes', [])
                    coords = []
                    for nid in node_ids:
                        if nid in nodes:
                            n = nodes[nid]
                            coords.append((n['lon'], n['lat']))
                    if len(coords) >= 3:
                        x_coords = [transformer_to_5179.transform(lon, lat)[0] for lon, lat in coords]
                        y_coords = [transformer_to_5179.transform(lon, lat)[1] for lon, lat in coords]
                        poly = Polygon(zip(x_coords, y_coords))
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        facilities.append({
                            'geometry': poly,
                            'facility_type': amenity,
                            'name': tags.get('name', tags.get('name:ko', '')),
                        })

            if not facilities:
                return _empty_gdf()

            gdf = gpd.GeoDataFrame(facilities, crs=f"EPSG:{EPSG_5179}")
            return gdf
        except Exception:
            return _empty_gdf()

    def _parse_subway_stations(self, osm_data: dict) -> list:
        """지하철역 파싱"""
        try:
            if not osm_data or 'elements' not in osm_data:
                return []

            stations = []
            for elem in osm_data['elements']:
                if elem['type'] != 'node':
                    continue
                tags = elem.get('tags', {})
                railway = tags.get('railway', '')
                subway = tags.get('subway', '')

                if railway in ('station', 'subway_entrance') or subway == 'yes':
                    lon, lat = elem.get('lon', 0), elem.get('lat', 0)
                    x, y = transformer_to_5179.transform(lon, lat)
                    stations.append({
                        'name': tags.get('name', tags.get('name:ko', '지하철역')),
                        'lon': lon,
                        'lat': lat,
                        'x': x,
                        'y': y,
                        'line': tags.get('line', ''),
                    })
            return stations
        except Exception:
            return []

    def _parse_bus_stops(self, osm_data: dict) -> list:
        """버스 정류장 파싱"""
        try:
            if not osm_data or 'elements' not in osm_data:
                return []

            stops = []
            for elem in osm_data['elements']:
                if elem['type'] != 'node':
                    continue
                tags = elem.get('tags', {})
                if tags.get('highway') == 'bus_stop':
                    lon, lat = elem.get('lon', 0), elem.get('lat', 0)
                    x, y = transformer_to_5179.transform(lon, lat)
                    stops.append({
                        'name': tags.get('name', tags.get('name:ko', '버스정류장')),
                        'lon': lon,
                        'lat': lat,
                        'x': x,
                        'y': y,
                    })
            return stops
        except Exception:
            return []

    def get_entry_points(
        self,
        boundary_geojson: dict,
        roads_gdf: gpd.GeoDataFrame
    ) -> list:
        """
        대상지 경계와 교차하는 기존 도로 탐색
        반환:
        [
          {
            'point': (x, y),        # EPSG:5179
            'point_wgs84': (lon,lat),
            'road_name': str,
            'road_width_m': float,
            'road_type': str,
            'angle_deg': float
          }
        ]
        """
        try:
            geom = shape(boundary_geojson)
            coords = list(geom.exterior.coords)
            transformed = [transformer_to_5179.transform(lon, lat) for lon, lat in coords]
            boundary_5179 = Polygon(transformed)
            boundary_line = boundary_5179.exterior

            if roads_gdf.empty:
                return []

            entry_points = []

            for idx, row in roads_gdf.iterrows():
                road_geom = row.geometry
                if road_geom is None:
                    continue

                intersection = boundary_line.intersection(road_geom)
                if intersection.is_empty:
                    continue

                pts = []
                if intersection.geom_type == 'Point':
                    pts = [intersection]
                elif intersection.geom_type == 'MultiPoint':
                    pts = list(intersection.geoms)
                elif intersection.geom_type == 'LineString':
                    centroid = intersection.centroid
                    pts = [centroid]
                elif intersection.geom_type == 'GeometryCollection':
                    for g in intersection.geoms:
                        if g.geom_type == 'Point':
                            pts.append(g)
                        elif g.geom_type == 'LineString':
                            pts.append(g.centroid)

                for pt in pts:
                    x, y = pt.x, pt.y
                    lon, lat = transformer_to_wgs84.transform(x, y)

                    # Compute angle from road direction at intersection
                    angle_deg = 0.0
                    try:
                        if isinstance(road_geom, LineString):
                            coords_road = list(road_geom.coords)
                            if len(coords_road) >= 2:
                                dx = coords_road[-1][0] - coords_road[0][0]
                                dy = coords_road[-1][1] - coords_road[0][1]
                                angle_deg = math.degrees(math.atan2(dy, dx)) % 360
                    except Exception:
                        pass

                    highway = row.get('highway', 'unclassified')
                    entry_points.append({
                        'point': (x, y),
                        'point_wgs84': (lon, lat),
                        'road_name': row.get('name', ''),
                        'road_width_m': float(row.get('width_m', 8)),
                        'road_type': highway,
                        'angle_deg': angle_deg,
                    })

            # Deduplicate nearby entry points (within 20m)
            deduplicated = []
            for ep in entry_points:
                px, py = ep['point']
                too_close = False
                for dep in deduplicated:
                    dx, dy = dep['point']
                    if math.hypot(px - dx, py - dy) < 20:
                        too_close = True
                        break
                if not too_close:
                    deduplicated.append(ep)

            return deduplicated

        except Exception:
            return []

    def analyze_surroundings(
        self,
        boundary_geojson: dict,
        roads_gdf: gpd.GeoDataFrame,
        landuse_gdf: gpd.GeoDataFrame,
        facilities_gdf: gpd.GeoDataFrame,
        subway_stations: list,
    ) -> dict:
        """
        반환:
        {
          'adjacent_use_north': str,
          'adjacent_use_south': str,
          'adjacent_use_east': str,
          'adjacent_use_west': str,
          'nearest_subway_m': float,
          'nearest_subway_name': str,
          'is_tod_zone': bool,        # 역세권 500m 이내
          'avg_building_height_m': float,
          'dominant_surrounding_use': str,
          'recommendations': list[str]
        }
        """
        try:
            geom = shape(boundary_geojson)
            coords = list(geom.exterior.coords)
            transformed = [transformer_to_5179.transform(lon, lat) for lon, lat in coords]
            boundary_5179 = Polygon(transformed)
            centroid = boundary_5179.centroid
            cx, cy = centroid.x, centroid.y

            # Analyze adjacent uses by quadrant
            def get_direction_use(direction: str) -> str:
                if landuse_gdf.empty:
                    return '미확인'
                offset = 200  # 200m offset in direction
                if direction == 'north':
                    probe = Point(cx, cy + offset)
                elif direction == 'south':
                    probe = Point(cx, cy - offset)
                elif direction == 'east':
                    probe = Point(cx + offset, cy)
                else:  # west
                    probe = Point(cx - offset, cy)

                min_dist = float('inf')
                closest_use = '미확인'
                for idx, row in landuse_gdf.iterrows():
                    d = probe.distance(row.geometry)
                    if d < min_dist:
                        min_dist = d
                        lu = row.get('landuse', '미확인')
                        closest_use = _translate_landuse(lu)
                return closest_use

            adjacent_north = get_direction_use('north')
            adjacent_south = get_direction_use('south')
            adjacent_east = get_direction_use('east')
            adjacent_west = get_direction_use('west')

            # Nearest subway
            nearest_subway_m = float('inf')
            nearest_subway_name = '없음'
            for station in subway_stations:
                sx, sy = station['x'], station['y']
                dist = math.hypot(cx - sx, cy - sy)
                if dist < nearest_subway_m:
                    nearest_subway_m = dist
                    nearest_subway_name = station['name']

            if nearest_subway_m == float('inf'):
                nearest_subway_m = 9999.0

            is_tod_zone = nearest_subway_m <= 500

            # Dominant surrounding use
            uses = [adjacent_north, adjacent_south, adjacent_east, adjacent_west]
            use_counts = {}
            for u in uses:
                if u != '미확인':
                    use_counts[u] = use_counts.get(u, 0) + 1
            dominant = max(use_counts, key=use_counts.get) if use_counts else '미확인'

            # Recommendations
            recommendations = []
            if is_tod_zone:
                recommendations.append(f"역세권({nearest_subway_name} {nearest_subway_m:.0f}m): 고밀 복합개발 권장")
            if dominant == '주거':
                recommendations.append("주변 주거지역 연속성 확보: 중저층 주거지역 배치 권장")
            elif dominant == '상업':
                recommendations.append("주변 상업지역 연계: 근린상업 또는 일반상업 배치 권장")
            if adjacent_north == '주거':
                recommendations.append("북측 주거지역 일조 고려: 북측 저층화 필요")

            return {
                'adjacent_use_north': adjacent_north,
                'adjacent_use_south': adjacent_south,
                'adjacent_use_east': adjacent_east,
                'adjacent_use_west': adjacent_west,
                'nearest_subway_m': round(nearest_subway_m, 1),
                'nearest_subway_name': nearest_subway_name,
                'is_tod_zone': is_tod_zone,
                'avg_building_height_m': 15.0,  # Default estimate
                'dominant_surrounding_use': dominant,
                'recommendations': recommendations,
            }

        except Exception:
            return {
                'adjacent_use_north': '미확인',
                'adjacent_use_south': '미확인',
                'adjacent_use_east': '미확인',
                'adjacent_use_west': '미확인',
                'nearest_subway_m': 9999.0,
                'nearest_subway_name': '없음',
                'is_tod_zone': False,
                'avg_building_height_m': 15.0,
                'dominant_surrounding_use': '미확인',
                'recommendations': [],
            }


def _translate_landuse(osm_landuse: str) -> str:
    """OSM landuse → 한국어 용도"""
    mapping = {
        'residential': '주거',
        'commercial': '상업',
        'retail': '상업',
        'industrial': '공업',
        'farmland': '농지',
        'forest': '임야',
        'grass': '녹지',
        'park': '공원',
        'recreation_ground': '공원',
        'school': '학교',
        'hospital': '의료',
        'government': '공공',
        'military': '군사',
        'cemetery': '묘지',
        'construction': '개발중',
        'brownfield': '나대지',
        'greenfield': '녹지',
    }
    return mapping.get(osm_landuse, osm_landuse if osm_landuse else '미확인')
