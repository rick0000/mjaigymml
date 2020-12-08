from mjaigym_ml.models.model import Head34Value1Model, Head2Model
from mjaigym_ml.config.model_config import ModelConfig


class ModelFactory:
    @classmethod
    def get_model(
            cls,
            model_type: str,
            model_config: ModelConfig,
            reach_dahai_feature_length: int,
            pon_chi_kan_feature_length: int,
    ):

        if model_type not in [
            "dahai", "reach", "pon", "kan", "chi"
        ]:
            raise Exception("invalid model type inputed.")

        if model_type == "dahai":
            return cls._get_dahai_model(
                model_config,
                reach_dahai_feature_length)
        elif model_type == "reach":
            return cls._get_reach_model(
                model_config,
                reach_dahai_feature_length)
        elif model_type == "pon":
            return cls._get_pon_model(
                model_config,
                pon_chi_kan_feature_length)
        elif model_type == "chi":
            return cls._get_chi_model(
                model_config,
                pon_chi_kan_feature_length)
        elif model_type == "kan":
            return cls._get_kan_model(
                model_config,
                pon_chi_kan_feature_length)

        raise Exception("not intended path")

    @classmethod
    def _get_dahai_model(cls, model_config: ModelConfig, feature_length: int):
        return Head34Value1Model(
            feature_length,
            model_config.mid_channels,
            model_config.resnet_repeat,
            model_config.learning_rate,
            model_config.batch_size
        )

    @classmethod
    def _get_reach_model(cls, model_config, feature_length):
        return Head2Model(
            feature_length,
            model_config.mid_channels,
            model_config.resnet_repeat,
            model_config.learning_rate,
            model_config.batch_size
        )

    @classmethod
    def _get_pon_model(cls, model_config, feature_length):
        return Head2Model(
            feature_length,
            model_config.mid_channels,
            model_config.resnet_repeat,
            model_config.learning_rate,
            model_config.batch_size
        )

    @classmethod
    def _get_kan_model(cls, model_config, feature_length):
        return Head2Model(
            feature_length,
            model_config.mid_channels,
            model_config.resnet_repeat,
            model_config.learning_rate,
            model_config.batch_size
        )

    @classmethod
    def _get_chi_model(cls, model_config, feature_length):
        return Head2Model(
            feature_length,
            model_config.mid_channels,
            model_config.resnet_repeat,
            model_config.learning_rate,
            model_config.batch_size
        )
