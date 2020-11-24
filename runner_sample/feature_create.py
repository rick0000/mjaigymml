
def run(extractor_config, input_dir, output_dir):
    extractor = feature_extractor(extractor_config)
    mjson_storage = MjsonStorage(input_dir)
    feature_storage = FeatureStorage(output_dir)
    
    # analyze feature
    for mjson in mjson_storage.get_mjson():
        game_feature_analysis = extractor.extract(mjson)
        feature_storage.save(game_feature_analysis)




if __name__ == "__main__":
    run()