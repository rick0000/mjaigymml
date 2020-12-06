from mjaigym_ml.features.feature_analyser import FeatureAnalyser
from mjaigym_ml.config.extract_config import ExtractConfig


class FeatureAnalyzerFactory():
    @classmethod
    def get_analyzer(cls, model_type: str, extract_config: ExtractConfig):

        if model_type not in [
            "dahai", "reach", "pon", "kan", "chi"
        ]:
            raise Exception("invalid model type inputed.")

        if model_type == "dahai":
            return cls._get_dahai_analyzer(extract_config)
        elif model_type == "reach":
            return cls._get_reach_analyzer(extract_config)
        elif model_type == "pon":
            return cls._get_pon_analyzer(extract_config)
        elif model_type == "chi":
            return cls._get_chi_analyzer(extract_config)
        elif model_type == "kan":
            return cls._get_kan_analyzer(extract_config)

        raise Exception("not intended path")

    @classmethod
    def _get_dahai_analyzer(cls, extract_config):
        return FeatureAnalyser(extract_config)

    @classmethod
    def _get_reach_analyzer(cls, extract_config):
        pass

    @classmethod
    def _get_pon_analyzer(cls, extract_config):
        pass

    @classmethod
    def _get_kan_analyzer(cls, extract_config):
        pass

    @classmethod
    def _get_chi_analyzer(cls, extract_config):
        pass
