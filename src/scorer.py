import mir_eval
import numpy as np
from utils import weimar2hertz

#should use window size instead of hardcoded values!
def evaluate_sample(sample):
    labels, predictions = sample.labels, sample.predictions
    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(
        ref_time = np.arange(np.size(labels)) * (256 / 22050),
        ref_freq = weimar2hertz(labels),
        est_time = np.arange(np.size(labels)) * (256 / 22050),
        est_freq = weimar2hertz(predictions)
    )

    vr, vfa = mir_eval.melody.voicing_measures(ref_v, est_v)
    rpa = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)
    oa = mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)


    return {
        'Voicing Recall': vr,
        'Voicing False Alarm': vfa,
        'Raw Pitch Accuracy': rpa,
        'Raw Chroma Accuracy': rca,
        'Overall Accuracy': oa
    }
