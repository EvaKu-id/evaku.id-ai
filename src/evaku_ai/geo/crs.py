def utm_epsg(lon: float, lat: float) -> int:
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError(f"invalid coordinate: lon={lon}, lat={lat}")
    zone = min(int((lon + 180) // 6) + 1, 60)
    return (32700 if lat < 0 else 32600) + zone
