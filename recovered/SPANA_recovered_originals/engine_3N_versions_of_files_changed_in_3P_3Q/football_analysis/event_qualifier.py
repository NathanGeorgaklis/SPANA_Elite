"""
event_qualifier.py -- EventQualifier: thin, stable entry point that
FootballEventDetector calls at the end of every possession segment.

EVENT VOCABULARY / ARBITRATION TASK: qualify_release() now delegates to
event_arbitration.resolve_release() instead of calling shot_candidate.py's
evaluate_release() directly. This is the ONE integration change this task
makes at this layer (item #16's "strictly necessary integration code"):
evaluate_release() itself is untouched and is still called -- exactly
once, with the exact same arguments as before -- from inside
event_arbitration.py, which treats its result as one candidate's evidence
among PASS/CROSS/SHOT/DUEL/UNKNOWN rather than an automatic final answer.
Two new, optional, additive keyword arguments (seg_start,
prior_segment_evidence) let the arbitration layer read the PRECEDING
possession segment's own already-computed evidence (incoming ball speed,
segment length -- both already produced by event_classifier.py's
EventClassifier.classify_segment(), untouched) without recomputing it.
See event_arbitration.py's module docstring for the full design."""
from .event_arbitration import resolve_release
from .shot_candidate import evaluate_goal_followup


class EventQualifier:
    def qualify_release(self, ball_tracker, possession_frames, i_end, fps=25.0, log=None,
                         pitch_context=None, keeper_candidate=None, seg_start=None, prior_segment_evidence=None):
        return resolve_release(ball_tracker, possession_frames, i_end, fps=fps, log=log,
                                pitch_context=pitch_context, keeper_candidate=keeper_candidate,
                                seg_start=seg_start, prior_segment_evidence=prior_segment_evidence)

    def qualify_goal_followup(self, shot_event, ball_tracker, possession_frames, fps=25.0, log=None,
                               pitch_context=None):
        return evaluate_goal_followup(shot_event, ball_tracker, possession_frames, fps=fps, log=log,
                                       pitch_context=pitch_context)
