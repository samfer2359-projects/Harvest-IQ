def recommend_fertilizer(N, P, K, ph, moisture):
    """Simple rule-based fertilizer recommendation system."""
    
    # Check for pH first since it affects nutrient availability
    if ph < 5.5:
        return "Acidic Soil", "Apply Lime to neutralize acidity"
    elif ph > 8:
        return "Alkaline Soil", "Apply Gypsum or Organic Matter to balance pH"
    
    # Check moisture level after pH
    if moisture < 30:
        return "Low Moisture", "Increase irrigation or use mulch to retain moisture"
    
    # Check for nutrient deficiencies only if pH and moisture are optimal
    if N < 50:
        return "Nitrogen Deficient", "Apply Urea or Ammonium Sulfate (Rich in Nitrogen)"
    elif P < 40:
        return "Phosphorus Deficient", "Apply Single Super Phosphate (SSP) or DAP"
    elif K < 40:
        return "Potassium Deficient", "Apply Muriate of Potash (MOP) or Sulfate of Potash (SOP)"
    
    # If everything is fine, give a balanced recommendation
    return "Balanced Soil", "Maintain regular organic manure application"
