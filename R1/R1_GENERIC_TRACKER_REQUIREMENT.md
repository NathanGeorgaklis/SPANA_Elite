# SPANA ELITE
## GENERIC PLAYER TRACKER REQUIREMENT

The `player_tracker_rb/` code must be fully generic.

It must NOT contain production logic that is specific to:

- SOL #23;
- MAR #4;
- any shirt number;
- any club;
- any team colour;
- any benchmark clip;
- any specific frame number;
- any specific camera position;
- any specific stadium;
- any known distractor such as the royal-blue ball boy.

Those entities may appear only in:
- human-label files;
- benchmark fixtures;
- test data;
- evaluation reports;
- diagnostic notes.

They must never influence the generic tracking algorithm.

## TARGET PLAYER ABSTRACTION

The tracker should know only:

`TARGET_PLAYER`

The target must be initialized from generic seed information such as:

- seed frame;
- seed bounding box;
- optional human-selected target crop;
- optional future user-selected identity metadata.

The tracker should not need to know:
- the player's name;
- shirt number;
- team;
- position;
- benchmark identity.

## OTHER PEOPLE

Every other person in the scene should initially be treated generically as:

`PERSON_CANDIDATE`

They may later become temporary tracklets or competitors.

The tracker must not assume that another detected person is:
- teammate;
- opponent;
- spectator;
- referee;
- ball boy;

unless a separate validated classifier provides that information.

Even then, team/role classification must not be required for basic target identity tracking.

## GENERIC IDENTITY QUESTION

At every frame, the tracker should answer:

> Which current detection, if any, has sufficient evidence to represent the same physical person selected at initialization?

The decision should be based on generic evidence such as:

- appearance similarity;
- contrast against nearby competitors;
- spatial continuity;
- scale consistency;
- motion plausibility;
- temporal continuity;
- occlusion context;
- detection quality;
- candidate uniqueness.

It must not use benchmark-specific knowledge.

## GENERIC COMPETITOR HANDLING

Any nearby person can compete with the target identity.

This includes:

- same-team player;
- opponent;
- referee;
- substitute;
- spectator;
- ball boy;
- staff member;
- partially visible person.

A competitor does not need to have a persistent track ID before it can block a target match.

The tracker should ask:

> Is this candidate sufficiently like the target, and sufficiently more like the target than nearby alternatives?

## NO SHIRT-NUMBER DEPENDENCY

Shirt-number recognition must not be required for target tracking.

If number recognition is later added, it may be used only as optional supporting evidence.

The tracker must still work when:
- numbers are invisible;
- numbers are blurred;
- players turn away;
- two players share similar-looking digits;
- OCR fails.

## NO TEAM-COLOUR DEPENDENCY

Kit colour can be one appearance cue, but it must never uniquely establish identity.

Many players on the same team wear identical colours.

Colour can help reject obviously incompatible candidates, but cannot prove identity by itself.

## NO FRAME-SPECIFIC RULES

Never implement logic such as:

- "after frame 266";
- "near frame 273";
- "during the GOLD collision";
- "if target is on the left side here";
- "ignore this region in this clip."

If a benchmark exposes a failure, translate it into a general rule.

Example:

BAD:
> At frame 267, do not switch to the player in front.

GOOD:
> When the target becomes fully occluded by another person, do not confirm the occluder as the target unless independent appearance evidence supports that identity.

## GENERIC REACQUISITION

Reacquisition must work for any target player.

After loss or occlusion, the tracker should:

1. generate plausible candidates;
2. compare them to trusted target history;
3. compare them against nearby competitors;
4. apply stricter evidence than ordinary frame-to-frame association;
5. return AMBIGUOUS or LOST if evidence is insufficient.

Do not use known future position or benchmark labels.

## GENERIC APPEARANCE MEMORY

The target appearance bank must be built from trusted observations of whichever player the user selected.

It should not contain hard-coded references to any benchmark player.

The appearance bank should support:
- multiple poses;
- front/back views;
- scale changes;
- lighting changes;
- partial visibility.

Only trusted observations may update it.

## GENERIC OUTPUT

The tracker output should use neutral identifiers such as:

`target_id: "T001"`

and generic candidate/tracklet IDs.

Do not write:
- `SOL23`;
- `MAR4`;
- player names;
- club names

into production identity logic.

Benchmark reports may map:

`T001 -> SOL #23`

for human readability, but this mapping must live outside the tracker.

## BENCHMARKS MUST NOT LEAK INTO THE TRACKER

GOLD is a test of the generic tracker.

The tracker must not know it is processing GOLD.

Running GOLD should use exactly the same tracker code and decision logic as any future user-uploaded football clip.

The only differences should be inputs such as:
- video;
- seed frame;
- seed box;
- configuration.

## REQUIRED TEST PRINCIPLE

For every important tracking rule, create synthetic tests with neutral identities:

- TARGET;
- PERSON_A;
- PERSON_B;
- PERSON_C.

Do not build core unit tests around SOL #23 and MAR #4.

GOLD remains a real-footage integration benchmark only.

## FUTURE PRODUCT REQUIREMENT

SPANA Elite must eventually support:

> User selects any player in any football video → tracker follows that physical player.

Therefore every architectural decision must be evaluated against:

> Would this still work if the selected player were someone completely different in another match?

If the answer is no, redesign it.

## FINAL RULE

SOL #23 and MAR #4 are benchmark subjects.

They are not part of the SPANA Elite tracking algorithm.

The production tracker must understand:

TARGET PLAYER

versus

OTHER PHYSICAL PEOPLE

and nothing benchmark-specific beyond that.