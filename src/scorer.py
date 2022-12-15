import mir_eval
import numpy as np
from utils import weimar2hertz


def strip(labels, predictions):
    start = np.nonzero(labels)[0].min()
    stop = np.nonzero(labels)[0].max()
    return labels[start:stop], predictions[start:stop]


def evaluate_sample(sample, allow_negative_predictions = False):
    labels, predictions = sample.pitch_sequence, sample.predictions
    if not allow_negative_predictions:
        predictions[predictions < 0] = 0
    labels, predictions = strip(labels, predictions)
    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(
        ref_time = np.arange(np.size(labels)) * sample.window,
        ref_freq = weimar2hertz(labels),
        est_time = np.arange(np.size(labels)) * sample.window,
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
