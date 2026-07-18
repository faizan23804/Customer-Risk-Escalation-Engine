import os
import sys
import joblib # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import torch # type: ignore
import warnings

warnings.filterwarnings('ignore')

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertModel # type: ignore
import shap # type: ignore

from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging
from Customer_Risk_Escalation.constant import *


# ── Typed output container ────────────────────────────────
@dataclass
class PredictionResult:
    risk_score   : float
    risk_level   : str
    escalated    : bool
    top_reasons  : List[Dict]
    model_used   : str = "LateFusion_LightGBM"


class PredictPipeline:

    def __init__(self):
        try:
            logging.info("Initializing PredictPipeline")
            (
                self.fusion_model,
                self.scaler,
                self.label_encoders,
                self.tokenizer,
                self.bert_model,
                self.device
            ) = self.load_artifacts()

            logging.info("PredictPipeline initialized")

        except Exception as e:
            raise CustomException(e, sys)


    def load_artifacts(self):
        try:
            # ── Find latest artifact directory ────────────
            artifact_base = ARTIFACTS_DIR
            runs = sorted(os.listdir(artifact_base), reverse=True)
            latest_run    = runs[0]
            artifact_path = os.path.join(artifact_base, latest_run)

            logging.info(f"Loading artifacts from: {artifact_path}")

            # ── Load fusion model ──────────────────────────
            fusion_model_path = os.path.join(
                artifact_path,
                FUSION_DIR_NAME,
                FUSION_MODEL_DIR,
                FUSION_MODEL_FILE_NAME
            )
            fusion_model = joblib.load(fusion_model_path)
            logging.info("Fusion model loaded")

            # ── Load scaler ────────────────────────────────
            scaler_path = os.path.join(
                artifact_path,
                DATA_TRANSFORMATION_DIR_NAME,
                DATA_TRANSFORMATION_SCALER_DIR,
                SCALER_FILE_NAME
            )
            scaler = joblib.load(scaler_path)
            logging.info("Scaler loaded")

            # ── Load label encoders ────────────────────────
            label_encoder_path = os.path.join(
                artifact_path,
                DATA_TRANSFORMATION_DIR_NAME,
                LABEL_ENCODER_DIR,
                LABEL_ENCODER_FILE
            )
            label_encoders = joblib.load(label_encoder_path)
            logging.info(f"Label encoders loaded: {list(label_encoders.keys())}")

            # ── Load DistilBERT ────────────────────────────
            device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
            bert_model = DistilBertModel.from_pretrained(DISTILBERT_MODEL_NAME)
            bert_model = bert_model.to(device)
            bert_model.eval()
            logging.info(f"DistilBERT loaded on {device}")

            return fusion_model, scaler, label_encoders, tokenizer, bert_model, device

        except Exception as e:
            raise CustomException(e, sys)


    def transform_input(self, raw_ticket: dict) -> pd.DataFrame:
        try:
            df = pd.DataFrame([raw_ticket])

            # ── Step 1: Handle missing values ─────────────────
            df['is_unresolved'] = int(
                pd.isnull(raw_ticket.get('resolution_time_hours'))
            )
            df['resolution_time_hours'] = df['resolution_time_hours'].fillna(
                SENTINEL_VALUE
            )

            # ── Step 2: Datetime features ──────────────────────
            created_at      = pd.to_datetime(df['created_at'].values[0])
            df['created_year']  = created_at.year
            df['created_month'] = created_at.month
            df['weekday_num']   = created_at.dayofweek

            # ── Step 3: Ordinal encode priority ───────────────
            priority_map = {
                'low': 1, 'medium': 2, 'high': 3, 'urgent': 4
            }
            df['priority'] = df['priority'].map(priority_map).fillna(2)

            # ── Step 4: Label encode categoricals ─────────────
            cat_cols = [
                'customer_segment', 'product_area',
                'issue_type', 'sla_plan', 'created_month'
            ]
            for col in cat_cols:
                if col in self.label_encoders and col in df.columns:
                    try:
                        df[col] = self.label_encoders[col].transform(
                            df[col].astype(str)
                        )
                    except ValueError:
                        df[col] = 0
                        logging.warning(f"Unseen category in {col} — defaulted to 0")

            # ── Final step: enforce EXACT training columns ─────────
            TRAINING_COLUMNS = [
                'customer_segment',
                'product_area',
                'issue_type',
                'priority',
                'sla_plan',
                'resolution_time_hours',
                'reopened',
                'has_attachment',
                'is_unresolved',
                'created_year',
                'created_month',
                'weekday_num'
            ]

            # Add missing columns with 0
            for col in TRAINING_COLUMNS:
                if col not in df.columns:
                    df[col] = 0
                    logging.warning(f"Added missing column: {col}")

            # Keep ONLY training columns in exact order
            df = df[TRAINING_COLUMNS].copy()

            # Verify no strings remain
            assert df.select_dtypes(include=['object']).empty, \
                f"String columns still present: {df.select_dtypes(include=['object']).columns.tolist()}"

            # Scale
            scaler_columns = [
                'customer_segment', 'product_area', 'issue_type',
                'sla_plan', 'resolution_time_hours', 'has_attachment',
                'created_year', 'created_month', 'weekday_num', 'is_unresolved'
            ]

            df[scaler_columns] = pd.DataFrame(
                self.scaler.transform(df[scaler_columns]),
                columns = scaler_columns,
                index   = df.index
            )

            # Final verification
            print(f"Final X_tabular shape  : {df.shape}")
            print(f"Final X_tabular dtypes : {df.dtypes.unique()}") # type: ignore

            logging.info(f"Input transformed. Shape: {df.shape}")
            return df # type: ignore

        except Exception as e:
            raise CustomException(e, sys)


    def extract_text_embedding(self, raw_ticket: dict) -> np.ndarray:
        try:
            # ── Build combined text ────────────────────────
            combined_text = (
                str(raw_ticket.get('initial_message', ''))    + " " +
                str(raw_ticket.get('agent_first_reply', ''))  + " " +
                str(raw_ticket.get('resolution_summary', ''))
            ).strip()

            if not combined_text:
                combined_text = "no text available"

            # ── Tokenize ───────────────────────────────────
            encoding = self.tokenizer(
                combined_text,
                max_length     = NLP_MAX_LENGTH,
                padding        = 'max_length',
                truncation     = True,
                return_tensors = 'pt'
            )

            input_ids      = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # ── Extract CLS embedding ──────────────────────
            with torch.no_grad():
                outputs = self.bert_model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask
                )

            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embedding     = cls_embedding.cpu().numpy()

            logging.info(f"Embedding extracted. Shape: {embedding.shape}")
            return embedding

        except Exception as e:
            raise CustomException(e, sys)


    def get_risk_level(self, score: float) -> str:
        try:
            if score > 0.80:
                return "Critical"
            elif score > 0.60:
                return "High"
            elif score > 0.40:
                return "Medium"
            else:
                return "Low"
        except Exception as e:
            raise CustomException(e, sys)


    def explain_prediction(self,
                           X_fused: np.ndarray,
                           feature_names: list) -> List[Dict]:
        try:
            # ── TreeExplainer — optimized for LightGBM ────
            explainer   = shap.TreeExplainer(self.fusion_model)
            shap_values = explainer.shap_values(X_fused)

            # ── For binary classification take class 1 ────
            if isinstance(shap_values, list):
                sv = shap_values[1][0]   # class 1, first row
            else:
                sv = shap_values[0]

            # ── Pair feature names with impact values ──────
            feature_impacts = list(zip(feature_names, sv))

            # ── Sort by absolute impact descending ─────────
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

            # ── Return top 5 as readable dicts ────────────
            top_reasons = [
                {
                    "feature": name,
                    "impact" : f"{'+' if val > 0 else ''}{val:.3f}",
                    "direction": "increases risk" if val > 0 else "decreases risk"
                }
                for name, val in feature_impacts[:5]
            ]

            tabular_reasons = [
                r for r in top_reasons 
                if not r['feature'].startswith('emb_')
            ] 

            logging.info(f"SHAP explanation computed. Top feature: {top_reasons[0]['feature']}")
            return tabular_reasons if tabular_reasons else top_reasons

        except Exception as e:
            raise CustomException(e, sys)


    def predict(self, raw_ticket: dict) -> PredictionResult:
        try:
            logging.info("="*50)
            logging.info("Prediction started")

            # ── Step 1: Transform tabular input ───────────
            X_tabular = self.transform_input(raw_ticket)

            # ── Step 2: Extract text embedding ────────────
            X_embedding = self.extract_text_embedding(raw_ticket)
            

            # ── Step 3: Concatenate ────────────────────────
            X_fused = np.concatenate(
                [X_tabular.values, X_embedding], axis=1
            ).astype(np.float64)   # ← force float64

            print(f"X_fused shape : {X_fused.shape}")
            print(f"X_fused dtype : {X_fused.dtype}")


            # ── Step 4: Predict probability ───────────────
            risk_score = float(
                self.fusion_model.predict_proba(X_fused)[0][1]
            )

            # ── Step 5: Get risk level ─────────────────────
            risk_level = self.get_risk_level(risk_score)

            # ── Step 6: Build feature names for SHAP ──────
            tabular_features   = X_tabular.columns.tolist()
            embedding_features = [f"emb_{i}" for i in range(X_embedding.shape[1])]
            all_feature_names  = tabular_features + embedding_features

            # ── Step 7: SHAP explanation ───────────────────
            top_reasons = self.explain_prediction(X_fused, all_feature_names)

            # ── Step 8: Build result ───────────────────────
            result = PredictionResult(
                risk_score  = round(risk_score, 4),
                risk_level  = risk_level,
                escalated   = risk_score > 0.50,
                top_reasons = top_reasons
            )

            logging.info(f"Prediction complete — Score: {risk_score:.4f} | Level: {risk_level}")
            logging.info("="*50)

            return result

        except Exception as e:
            raise CustomException(e, sys)


# ── Quick test ─────────────────────────────────────────────
if __name__ == "__main__":
    sample_ticket = {
        "customer_segment"     : "enterprise",
        "product_area"         : "billing",
        "issue_type"           : "account_access",
        "priority"             : "high",
        "status"               : "open",
        "sla_plan"             : "premium",
        "initial_message"      : "I cannot access my account and nobody is helping me.",
        "agent_first_reply"    : "We are looking into this issue.",
        "resolution_summary"   : None,
        "resolution_time_hours": None,
        "reopened"             : 0,
        "customer_sentiment"   : "negative",
        "csat_score"           : 2,
        "has_attachment"       : 0,
        "platform"             : "web",
        "region"               : "EU",
        "created_at"           : "2026-07-15T10:30:00"
    }

    pipeline = PredictPipeline()
    result   = pipeline.predict(sample_ticket)

    print(f"\nRisk Score  : {result.risk_score}")
    print(f"Risk Level  : {result.risk_level}")
    print(f"Escalated   : {result.escalated}")
    print(f"\nTop Reasons:")
    for r in result.top_reasons:
        print(f"  {r['feature']:30s} {r['impact']:8s} ({r['direction']})")