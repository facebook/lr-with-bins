//Copyright (c) Meta Platforms, Inc. and affiliates.
//
//This source code is licensed under the MIT license found in the
//LICENSE file in the root directory of this source tree.

<?hh

abstract class BinaryClassificationMultistageInference{
  /**
   * Returns the combined bin key in which `$data` falls according to the
   * quantile bins found in `$quantiles`.
   */
  private function dataToCombinedBin(
    vec<vec<float>> $quantiles,
    vec<float> $data,
  ): string {
    $num_quantile_bins = C\count($quantiles);
    $num_features = C\count($quantiles[0]);

    $combined_bin = "";
    for ($j = 0; $j < $num_features; $j++) {
      $quantile_bin = 0;
      for ($i = 0; $i < $num_quantile_bins; $i++) {
        if ($data[$j] > $quantiles[$i][$j]) {
          $quantile_bin++;
        }
      }
      $combined_bin .= $quantile_bin;
    }
    return $combined_bin;
  }

  /**
   * Normalizes data to the combined bin metrics found in `$weights_info`. Then,
   * performs logistic regression inference on `$data`.
   */
  private function lrInference(
    dict<string, vec<float>> $weights_info,
    vec<float> $data,
  ): float {
    $mean = vec($weights_info["mean"]);
    $std = vec($weights_info["std"]);
    $eps = vec($weights_info["eps"]);
    $weights = vec($weights_info["weights"]);
    $len_data_row = C\count($data);
    for ($i = 0; $i < $len_data_row; $i++) {
      $data[$i] = ($data[$i] - $mean[$i]) / ($std[$i] + $eps[0]);
    }
    $sum = 0.0;
    $sum += $weights[0];
    for ($i = 0; $i < $len_data_row; $i++) {
      $sum += $data[$i] * $weights[$i + 1];
    }
    $prob = 1 / (1 + Math\exp(-$sum));
    return $prob;
  }

  /**
   * Return the features values (provided by `$id_to_feature_value`) of the
   * `$feature_ids` vector in that order. Features not found are set to zero.
   * Features are normalized according to `$data_mean`, `$data_std`, and `$eps`.
   */
  private function getData(
    dict<string, string> $id_to_feature_value,
    vec<string> $feature_ids,
    vec<float> $data_mean,
    vec<float> $data_std,
    float $eps,
  ): vec<float> {
    $num_features = C\count($feature_ids);
    $data = vec[];
    for ($i = 0; $i < $num_features; $i++) {
      if (C\contains_key($id_to_feature_value, $feature_ids[$i])) {
        $feature_val = (float)$id_to_feature_value[$feature_ids[$i]];
      } else {
        $feature_val = 0.0;
      }
      $data[] = ($feature_val - $data_mean[$i]) / ($data_std[$i] + $eps);
    }
    return $data;
  }

  /**
   * This first stage inference is called first by genProbability to attempt a
   * quick and accurate inference using the LRwBins model. The function returns
   * the probability of the logistic regression model or -1 if first-stage
   * inference cannot handle the data.
   */
  private async function genFirstStageInference(
    this::TContext $context,
  ): Awaitable<?float> {
    // load in the first-stage inference model
    $config_name = "experimental/danielsjohnson/model";
    $raw_config = Configerator::getRaw($config_name);
    $json_a = fb_json_decode($raw_config, true);

    // access the loop features
    $feature_handler = await $this->genConfigBackedFeatureHandler();
    $features_wrapper = await $feature_handler->genFeaturesAndContext(
      new LooperMemoizingWrapper($context),
    );
    list($features, $_contexts) = $features_wrapper;
    $ids = $features->getIDs();

    // map feature ids to feature values
    $id_to_feature_value = dict[];
    foreach ($ids as $id) {
      $id_to_feature_value[(string)$id] =
        PHP\strval($features->getFeature($id));
    }

    // create vector of feature values in the order used to construct the
    // combined bins (the order found in `$bin_feature_ids`)
    $eps = (float)$json_a["eps"];
    $bin_feature_ids = vec($json_a["bin_feature_ids"]);
    $bin_data_mean = vec($json_a["bin_data_mean"]);
    $bin_data_std = vec($json_a["bin_data_std"]);
    $bin_data = $this->getData(
      $id_to_feature_value,
      $bin_feature_ids,
      $bin_data_mean,
      $bin_data_std,
      $eps,
    );

    // get the combined bin key of the combined bin that `$bin_data` falls into
    $quantiles = vec($json_a["quantiles"]);
    $combined_bin = $this->dataToCombinedBin($quantiles, $bin_data);

    // check if the first-stage inference model is used for this combined bin
    $combined_bin_weights = $json_a["combined_bins"];
    if (C\contains_key($combined_bin_weights, $combined_bin)) {
      // first-stage inference is used for this prediction
      // create vector of feature values in the same order as their corresponding
      // logistic regression weights (the order found in `$inference_feature_ids`)
      $inference_feature_ids = vec($json_a["inference_feature_ids"]);
      $inference_data_mean = vec($json_a["inference_data_mean"]);
      $inference_data_std = vec($json_a["inference_data_std"]);
      $inference_data = $this->getData(
        $id_to_feature_value,
        $inference_feature_ids,
        $inference_data_mean,
        $inference_data_std,
        $eps,
      );

      // access the combined bin weights and perform logistic regression
      // inference on `$inference_data`
      $prob = $this->lrInference(
        $combined_bin_weights[$combined_bin],
        $inference_data,
      );
    } else {
      // first-stage inference is not used for this combined bin
      $prob = -1.0;
    }
    return $prob;
  }
}

