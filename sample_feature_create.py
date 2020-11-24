
from mjaigym_ml.features.game_feature_analyser import GameFeatureAnalyser
from mjaigym_ml.features.game_feature_analysis import GameFeatureAnalysis
from mjaigym_ml.storage.local_file_mjson_storage import LocalFileMjsonStorage
from mjaigym_ml.storage.local_file_mjson_storage import LocalFileMjsonStorage


def run(extractor_config, input_dir, output_dir):
    analyser = GameFeatureAnalyser(extractor_config)
    mjson_storage = LocalFileMjsonStorage(input_dir)
    feature_storage = LocalFileMjsonStorage(output_dir)
    
    # save config

    # analyze feature
    for mjson in mjson_storage.get_mjsons():
        game_feature_analysis = analyser.analyse_mjson(mjson)
        labels, features = game_feature_analysis.get_records()
        # feature_storage.save(labels, features)


if __name__ == "__main__":
    input_dir = "sample_mjson"
    output_dir = "outputs"
    config = {
        "featureAAAv0_0_0":"FeatureAAAv0"
    }
    run(config, input_dir, output_dir)