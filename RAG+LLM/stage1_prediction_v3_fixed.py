"""
Stage 1 Prediction Pipeline - V3 Fixed
========================================

Corrected feature derivation based on proper feature engineering logic.
Simplified RAG output with only essential health context.

Key Fixes:
1. Proper cycle detection using phase transitions
2. Correct cycle length and variation calculation
3. Fixed days_since_last_period (no negatives)
4. Proper questionnaire value handling (no auto-zero)
5. Adjusted ensemble logic to respect medical rules
6. Simplified RAG output format

Author: HerWell Team
Date: March 2026
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "stage1_v3_trained_model"

MODEL_FILE = MODEL_DIR / "rf_model_v3.pkl"
SCALER_FILE = MODEL_DIR / "scaler_v3.pkl"
FEATURES_FILE = MODEL_DIR / "feature_names_v3.json"

ML_CONFIDENCE_THRESHOLD = 0.80  # Increased to 80% to respect medical rules more

# =============================================================================
# LOAD TRAINED MODEL & ARTIFACTS
# =============================================================================

print("="*80)
print("STAGE 1 PREDICTION PIPELINE - V3 FIXED")
print("="*80)

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

rf_model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

with open(FEATURES_FILE, 'r') as f:
    FEATURE_NAMES = json.load(f)

print(f"\n✓ Loaded models and {len(FEATURE_NAMES)} features")

# =============================================================================
# CYCLE FEATURE DERIVATION (CORRECTED)
# =============================================================================

def detect_cycle_start(group):
    """
    Detect menstrual cycle starts using phase transitions into 'Menstrual'.
    
    Based on feature engineering pipeline logic:
    - cycle_start_flag = 1 when current phase is Menstrual AND 
      previous day phase is not Menstrual (or missing)
    
    Returns group with:
    - cycle_start_flag
    - cycle_id
    """
    group = group.sort_values("day_in_study").copy()
    prev_phase = group["phase"].shift(1)
    
    # Detect cycle starts
    menstrual_today = group["phase"].eq("Menstrual")
    menstrual_yesterday = prev_phase.fillna(False).eq("Menstrual")
    
    group["cycle_start_flag"] = (menstrual_today & ~menstrual_yesterday).astype(int)
    
    # Assign cycle IDs
    if group["cycle_start_flag"].sum() == 0:
        # No cycles detected, assign all to cycle 0
        group["cycle_id"] = 0
    else:
        # Increment cycle_id at each cycle start
        group["cycle_id"] = group["cycle_start_flag"].cumsum()
    
    return group

def assign_cycle_length(group):
    """
    For each participant:
    - detect cycle starts
    - compute cycle length as days between consecutive cycle starts
    - assign the cycle length to all rows within that cycle
    
    Also computes:
    - days_since_last_period
    """
    group = group.sort_values("day_in_study").copy()
    
    # Detect cycle starts
    cycle_starts = group.loc[group["cycle_start_flag"] == 1, ["cycle_id", "day_in_study"]].copy()
    
    if cycle_starts.empty:
        group["cycle_length_estimate"] = np.nan
        group["days_since_last_period"] = np.nan
        return group
    
    # Compute cycle length for each cycle
    cycle_starts = cycle_starts.rename(columns={"day_in_study": "cycle_start_day"})
    cycle_starts["cycle_length_estimate"] = (
        cycle_starts["cycle_start_day"].shift(-1) - cycle_starts["cycle_start_day"]
    )
    
    # Merge cycle length back to all rows
    group = group.merge(
        cycle_starts[["cycle_id", "cycle_start_day", "cycle_length_estimate"]],
        on="cycle_id",
        how="left"
    )
    
    # Days since last period = current day - cycle start day
    group["days_since_last_period"] = group["day_in_study"] - group["cycle_start_day"]
    
    return group

def add_cycle_variation(group):
    """
    Compute cycle length variation across previous cycles for each participant.
    Uses rolling std over cycle lengths at the cycle level.
    """
    group = group.copy()
    
    # Get cycle-level data
    cycle_level = (
        group.groupby(["id", "cycle_id"], as_index=False)
        .agg(cycle_length_estimate=("cycle_length_estimate", "first"))
        .sort_values(["id", "cycle_id"])
    )
    
    if cycle_level.empty:
        group["cycle_length_variation"] = np.nan
        return group
    
    # Rolling std over cycle lengths
    cycle_level["cycle_length_variation"] = (
        cycle_level.groupby("id")["cycle_length_estimate"]
        .transform(lambda s: s.rolling(window=3, min_periods=2).std())
    )
    
    # Merge back
    group = group.merge(
        cycle_level[["id", "cycle_id", "cycle_length_variation"]],
        on=["id", "cycle_id"],
        how="left"
    )
    
    return group

def derive_cycle_features_from_daily_records(daily_records_df):
    """
    Extract cycle features from user's phase history.
    
    Follows the exact logic from feature engineering pipeline:
    1. Detect cycle starts (Menstrual phase transitions)
    2. Assign cycle IDs
    3. Compute cycle lengths
    4. Compute cycle variation
    5. Compute days since last period
    
    Returns:
    --------
    dict with:
        - cycle_length_estimate (mean of all cycles)
        - cycle_length_variation (recent variation)
        - days_since_last_period (from last cycle start to today)
    """
    
    if len(daily_records_df) == 0:
        return {
            'cycle_length_estimate': 28.0,
            'cycle_length_variation': 0.0,
            'days_since_last_period': 14.0
        }
    
    # Prepare data
    df = daily_records_df.copy()
    
    # Ensure Date is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date')
        # Create day_in_study from dates
        df['day_in_study'] = (df['Date'] - df['Date'].min()).dt.days + 1
    elif 'day_in_study' not in df.columns:
        # No date info available
        return {
            'cycle_length_estimate': 28.0,
            'cycle_length_variation': 0.0,
            'days_since_last_period': 14.0
        }
    
    # Must have 'phase' column
    if 'phase' not in df.columns:
        return {
            'cycle_length_estimate': 28.0,
            'cycle_length_variation': 0.0,
            'days_since_last_period': 14.0
        }
    
    # Detect cycles
    df = detect_cycle_start(df)
    
    # Assign cycle lengths
    df = assign_cycle_length(df)
    
    # Add cycle variation
    df = add_cycle_variation(df)
    
    # Get the most recent valid values
    # Filter to rows with non-null cycle_length_estimate
    valid_rows = df.dropna(subset=['cycle_length_estimate'])
    
    if len(valid_rows) == 0:
        # No valid cycles detected
        return {
            'cycle_length_estimate': 28.0,
            'cycle_length_variation': 0.0,
            'days_since_last_period': 14.0
        }
    
    # Take the most recent row with valid data
    latest = valid_rows.iloc[-1]
    
    # Extract values
    cycle_length_estimate = latest['cycle_length_estimate']
    cycle_length_variation = latest.get('cycle_length_variation', 0.0)
    if pd.isna(cycle_length_variation):
        cycle_length_variation = 0.0
    
    # Days since last period should never be negative
    days_since_last = latest.get('days_since_last_period', 14.0)
    if pd.isna(days_since_last) or days_since_last < 0:
        # If negative or null, compute from date
        if 'Date' in df.columns and 'cycle_start_day' in latest:
            last_cycle_start_date = df[df['day_in_study'] == latest['cycle_start_day']]['Date'].iloc[0]
            days_since_last = (pd.Timestamp.now() - last_cycle_start_date).days
            # Ensure positive
            days_since_last = max(0, days_since_last)
        else:
            days_since_last = 14.0
    
    return {
        'cycle_length_estimate': float(cycle_length_estimate),
        'cycle_length_variation': float(cycle_length_variation),
        'days_since_last_period': float(days_since_last)
    }

# =============================================================================
# FEATURE COMBINATION (CORRECTED)
# =============================================================================

def combine_features(questionnaire_data, daily_record_data):
    """
    Combine questionnaire + daily records into 17-feature prediction input.
    
    Key fixes:
    1. Only fill NaN with 0 if value is truly missing (NaN)
    2. Don't override actual 0 values from questionnaire
    3. Validate feature ranges
    
    Parameters:
    -----------
    questionnaire_data : DataFrame or dict
    daily_record_data : DataFrame
    
    Returns:
    --------
    DataFrame with exactly 17 features in correct order
    """
    
    # Convert questionnaire to dict
    if isinstance(questionnaire_data, pd.DataFrame):
        # Take the latest non-empty row
        # Find row with most non-null values
        non_null_counts = questionnaire_data.notna().sum(axis=1)
        best_row_idx = non_null_counts.idxmax()
        q_dict = questionnaire_data.loc[best_row_idx].to_dict()
    else:
        q_dict = questionnaire_data.copy()
    
    # Derive cycle features from daily records
    cycle_features = derive_cycle_features_from_daily_records(daily_record_data)
    
    # Build combined features dict
    combined = {}
    
    # Cycle features (from daily records)
    combined['cycle_length_estimate'] = cycle_features['cycle_length_estimate']
    combined['cycle_length_variation'] = cycle_features['cycle_length_variation']
    combined['days_since_last_period'] = cycle_features['days_since_last_period']
    
    # Helper function to get value or default
    def get_or_default(key, default=0.0):
        val = q_dict.get(key, default)
        # Only use default if value is NaN
        if pd.isna(val):
            return default
        return float(val)
    
    # Symptom features (from questionnaire)
    combined['bleeding_duration_days'] = get_or_default('bleeding_duration_days', 0)
    combined['pain_score'] = get_or_default('pain_score', 0)
    combined['pain_trend'] = get_or_default('pain_trend', 0)
    combined['headache_score'] = get_or_default('headache_score', 0)
    combined['fatigue_score'] = get_or_default('fatigue_score', 0)
    combined['sleep_issue_score'] = get_or_default('sleep_issue_score', 0)
    combined['mood_instability'] = get_or_default('mood_instability', 0)
    combined['stress_score'] = get_or_default('stress_score', 0)
    combined['bloating_score'] = get_or_default('bloating_score', 0)
    combined['symptom_burden_score'] = get_or_default('symptom_burden_score', 0)
    
    # Flow features (from questionnaire)
    combined['flow_volume_num'] = get_or_default('flow_volume_num', 0)
    combined['heavy_flow_flag'] = get_or_default('heavy_flow_flag', 0)
    
    # User demographics
    combined['age'] = get_or_default('age', 28)
    combined['sexual_activity_flag'] = get_or_default('sexual_activity_flag', 0)
    
    # Validate feature ranges
    combined = validate_features(combined)
    
    # Create DataFrame with exact feature order
    return pd.DataFrame([combined])[FEATURE_NAMES]

def validate_features(features):
    """
    Validate feature values to prevent impossible values.
    """
    # Cycle length: 15-60 days (extended range for abnormal cases)
    if features['cycle_length_estimate'] < 10:
        features['cycle_length_estimate'] = 10.0
    elif features['cycle_length_estimate'] > 90:
        features['cycle_length_estimate'] = 90.0
    
    # Days since period: 0-120 days
    if features['days_since_last_period'] < 0:
        features['days_since_last_period'] = 0.0
    elif features['days_since_last_period'] > 120:
        features['days_since_last_period'] = 120.0
    
    # All symptom scores: 0-10 range
    symptom_fields = ['pain_score', 'headache_score', 'fatigue_score', 
                      'sleep_issue_score', 'mood_instability', 'stress_score', 
                      'bloating_score', 'symptom_burden_score']
    for field in symptom_fields:
        if features[field] < 0:
            features[field] = 0.0
        elif features[field] > 10:
            features[field] = 10.0
    
    # Flow volume: 0-6
    if features['flow_volume_num'] < 0:
        features['flow_volume_num'] = 0.0
    elif features['flow_volume_num'] > 6:
        features['flow_volume_num'] = 6.0
    
    # Age: 18-50
    if features['age'] < 18:
        features['age'] = 18.0
    elif features['age'] > 50:
        features['age'] = 50.0
    
    return features

# =============================================================================
# RULE-BASED CLASSIFIER
# =============================================================================

def classify_risk_rules(user_data):
    """
    Apply rule-based classification.
    
    Returns: (risk_level, rule_trigger)
    """
    
    # RISK 3: EMERGENCY
    if user_data.get('pain_score', 0) >= 8 and user_data.get('heavy_flow_flag', 0) == 1:
        return (3, 'severe_pain_heavy_bleeding')
    
    if user_data.get('pain_score', 0) >= 8 and user_data.get('symptom_burden_score', 0) >= 4:
        return (3, 'severe_pain_systemic_symptoms')
    
    # RISK 2: URGENT CONSULTATION
    if user_data.get('days_since_last_period', 0) > 90:
        return (2, 'amenorrhea')
    
    cycle_len = user_data.get('cycle_length_estimate')
    if cycle_len is not None and not pd.isna(cycle_len):
        if cycle_len < 21 or cycle_len > 35:
            return (2, 'abnormal_cycle_length')
    
    if user_data.get('bleeding_duration_days', 0) > 7:
        return (2, 'prolonged_menstrual_bleeding')
    
    if user_data.get('pain_score', 0) >= 5:
        return (2, 'moderate_severe_dysmenorrhea')
    
    if user_data.get('pain_score', 0) >= 4 and user_data.get('pain_trend', 0) > 0.5:
        return (2, 'worsening_menstrual_pain')
    
    # RISK 1: MONITOR
    pain = user_data.get('pain_score', 0)
    if 1 <= pain <= 2:
        return (1, 'mild_dysmenorrhea')
    
    symptom_burden = user_data.get('symptom_burden_score', 0)
    if 2 <= symptom_burden < 3.5:
        return (1, 'moderate_pms_symptom_burden')
    
    flow_num = user_data.get('flow_volume_num', 0)
    if 2 <= flow_num <= 3:
        return (1, 'moderate_menstrual_flow')
    
    if user_data.get('stress_score', 0) >= 3 or user_data.get('fatigue_score', 0) >= 3:
        return (1, 'stress_or_fatigue_signals')
    
    # RISK 0: NORMAL
    return (0, 'normal')

# =============================================================================
# ML CLASSIFIER
# =============================================================================

def classify_risk_ml(user_features_df):
    """
    Apply ML-based classification.
    
    Returns: (risk_level, confidence)
    """
    X = user_features_df[FEATURE_NAMES].values
    X_scaled = scaler.transform(X)
    
    risk_pred = rf_model.predict(X_scaled)[0]
    risk_proba = rf_model.predict_proba(X_scaled)[0]
    
    return (int(risk_pred), float(risk_proba[risk_pred]))

# =============================================================================
# ENSEMBLE DECISION LOGIC (ADJUSTED)
# =============================================================================

def ensemble_decision(rule_risk, ml_risk, ml_confidence, threshold=ML_CONFIDENCE_THRESHOLD):
    """
    Adjusted ensemble logic to respect medical rules.
    
    Decision Logic:
    1. Emergency (rule_risk == 3) → ALWAYS use rules
    2. Urgent medical flag (rule_risk == 2) → Use rules UNLESS ML has very high confidence (>80%)
    3. Monitor/Normal → Use ML if confidence >= threshold
    4. Otherwise → Use rules (conservative)
    
    This ensures medical guidelines are respected while allowing ML to assist.
    """
    
    # Safety: Always trust emergency rules
    if rule_risk == 3:
        return (3, 'rules_emergency')
    
    # Medical concern: Only override if ML is very confident
    if rule_risk == 2:
        if ml_confidence >= 0.85:  # Very high bar for overriding urgent medical flag
            return (ml_risk, 'ml_very_high_confidence')
        else:
            return (2, 'rules_medical_concern')
    
    # For Risk 0 or 1 from rules
    if ml_confidence >= threshold:
        return (ml_risk, 'ml_high_confidence')
    else:
        return (rule_risk, 'rules_low_ml_confidence')

# =============================================================================
# MAIN PREDICTION FUNCTION (SIMPLIFIED RAG OUTPUT)
# =============================================================================

def predict(questionnaire_data, daily_record_data, verbose=True):
    """
    Complete Stage 1 prediction from raw user inputs.
    
    Returns simplified JSON for RAG integration.
    """
    
    if verbose:
        print("\n" + "="*80)
        print("PROCESSING USER INPUT")
        print("="*80)
    
    # Step 1: Combine features
    user_features_df = combine_features(questionnaire_data, daily_record_data)
    
    if verbose:
        print(f"\n✓ Derived 17 features:")
        for feat, val in user_features_df.iloc[0].items():
            print(f"   {feat:30s} = {val:.2f}")
    
    # Step 2: Rule-based classification
    user_data_dict = user_features_df.iloc[0].to_dict()
    rule_risk, rule_trigger = classify_risk_rules(user_data_dict)
    
    if verbose:
        print(f"\n🔧 Rule prediction: Risk {rule_risk} ({rule_trigger})")
    
    # Step 3: ML classification
    ml_risk, ml_confidence = classify_risk_ml(user_features_df)
    
    if verbose:
        print(f"🤖 ML prediction: Risk {ml_risk} (confidence: {ml_confidence:.1%})")
    
    # Step 4: Ensemble decision
    final_risk, decision_source = ensemble_decision(
        rule_risk, ml_risk, ml_confidence, ML_CONFIDENCE_THRESHOLD
    )
    
    if verbose:
        print(f"⚖️  Final decision: Risk {final_risk} (source: {decision_source})")
        if rule_risk != ml_risk:
            print(f"   ⚠️  Disagreement: Rules={rule_risk}, ML={ml_risk}")
    
    # Step 5: Build simplified RAG output
    risk_labels = {
        0: "Normal",
        1: "Monitor",
        2: "Urgent Consultation",
        3: "Emergency"
    }
    
    # Create clean user features dict for RAG
    user_features_clean = {
        # Demographics
        'age': int(user_data_dict['age']),
        'sexual_activity': bool(user_data_dict['sexual_activity_flag']),
        
        # Cycle info
        'cycle_length': round(user_data_dict['cycle_length_estimate'], 1),
        'cycle_variation': round(user_data_dict['cycle_length_variation'], 1),
        'days_since_last_period': int(user_data_dict['days_since_last_period']),
        
        # Flow
        'bleeding_duration_days': int(user_data_dict['bleeding_duration_days']),
        'flow_volume': int(user_data_dict['flow_volume_num']),
        'heavy_flow': bool(user_data_dict['heavy_flow_flag']),
        
        # Symptoms
        'pain_score': int(user_data_dict['pain_score']),
        'pain_trend': round(user_data_dict['pain_trend'], 2),
        'headache_score': int(user_data_dict['headache_score']),
        'fatigue_score': int(user_data_dict['fatigue_score']),
        'sleep_issue_score': int(user_data_dict['sleep_issue_score']),
        'mood_instability': int(user_data_dict['mood_instability']),
        'stress_score': int(user_data_dict['stress_score']),
        'bloating_score': int(user_data_dict['bloating_score']),
        'symptom_burden_score': round(user_data_dict['symptom_burden_score'], 1)
    }
    
    # Return simplified structure for RAG
    return {
        'risk_assessment': {
            'risk_level': final_risk,
            'risk_label': risk_labels[final_risk]
        },
        'user_features': user_features_clean
    }

# =============================================================================
# DEMO / TEST SECTION
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("DEMO: TESTING WITH REAL USER DATA")
    print("="*80)
    
    try:
        questionnaire = pd.read_csv("Questionnaire_Data.csv")
        daily_records = pd.read_csv("Daily_Record_Test_1.csv")
        
        print(f"\n✓ Loaded questionnaire: {questionnaire.shape}")
        print(f"✓ Loaded daily records: {daily_records.shape}")
        
        # Run prediction
        result = predict(questionnaire, daily_records, verbose=True)
        
        # Display final result
        print("\n" + "="*80)
        print("📊 FINAL PREDICTION RESULT")
        print("="*80)
        
        print(f"\n🎯 Risk Level: {result['risk_assessment']['risk_level']}")
        print(f"📋 Risk Label: {result['risk_assessment']['risk_label']}")
        
        print(f"\n👤 User Features:")
        for key, value in result['user_features'].items():
            print(f"   {key:25s} = {value}")
        
        # Show simplified JSON
        print("\n" + "="*80)
        print("📦 SIMPLIFIED RAG OUTPUT (JSON)")
        print("="*80)
        print(json.dumps(result, indent=2))
        
        # Save output
        output_file = Path("stage1_v3_trained_model") / "rag_output_simplified.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  Test data not found: {e}")
