"""Regression tests for complementary PPE fusion and disagreement handling."""
from ppe_classes import CLASS_IDS
from ppe_types import Detection
from ppe_ensemble import combine_ppe
from ppe_rules import RuleEngine
from ppe_config import ModelConfig, RulesConfig

PERSON=(200.,60.,440.,440.)
TORSO=(230.,160.,400.,330.)


def d(name,box=TORSO,conf=.8):
    return Detection(CLASS_IDS[name],name,conf,box)


def status(dets):
    obs=RuleEngine(RulesConfig(required_ppe=['safety-vest']),ModelConfig()).evaluate_frame(dets,(640,480))
    return obs[0].items['safety-vest'].status


def test_agreed_vest_uses_weaker_score_without_confidence_inflation():
    result=combine_ppe([d('safety-vest',conf=.9)],[d('safety-vest',conf=.4)])
    assert len(result)==1 and result[0].name=='safety-vest'
    assert result[0].conf==.4


def test_one_model_vest_is_unknown_instead_of_worn_or_missing():
    result=combine_ppe([d('safety-vest')],[d('person',PERSON)])
    assert any(x.name=='uncertain-vest' for x in result)
    assert not any(x.name=='safety-vest' for x in result)
    assert status(result)=='UNKNOWN'


def test_reference_only_vest_is_also_unknown():
    result=combine_ppe([],[d('person',PERSON),d('safety-vest')])
    assert status(result)=='UNKNOWN'


def test_nonoverlapping_vest_votes_are_not_agreement():
    result=combine_ppe([d('safety-vest',(0.,0.,50.,80.))],[d('safety-vest',(100.,0.,150.,80.))])
    assert len(result)==2 and all(x.name=='uncertain-vest' for x in result)


def test_one_reference_vote_cannot_confirm_two_vests():
    result=combine_ppe([d('safety-vest'),d('safety-vest',(235.,165.,405.,335.))],[d('safety-vest')])
    assert sum(x.name=='safety-vest' for x in result)==1
    assert sum(x.name=='uncertain-vest' for x in result)==1


def test_original_people_and_bare_head_evidence_are_preserved():
    candidate=d('person',(0.,0.,100.,200.),.95)
    person=d('person',PERSON)
    head=d('head',(270.,70.,330.,130.))
    result=combine_ppe([candidate],[person,head])
    assert candidate not in result and person in result and head in result


def test_helmet_and_no_helmet_are_distinct_and_duplicates_are_suppressed():
    result=combine_ppe([d('helmet',conf=.7),d('no-helmet')],[d('helmet',conf=.8)])
    assert sum(x.name=='helmet' for x in result)==1
    assert sum(x.name=='no-helmet' for x in result)==1


def test_disagreement_blocks_explicit_missing_vest_accusation():
    result=combine_ppe([d('safety-vest'),d('no-vest',conf=.9)],[d('person',PERSON)])
    assert status(result)=='UNKNOWN'


def test_agreed_presence_wins_over_extra_unmatched_vote():
    result=combine_ppe([d('safety-vest'),d('safety-vest',(235.,165.,405.,335.))],
                       [d('person',PERSON),d('safety-vest')])
    assert status(result)=='PRESENT'


def test_neither_model_voting_is_not_automatic_presence():
    result=combine_ppe([],[d('person',PERSON)])
    assert status(result)=='MISSING_ABSENT'  # existing stricter temporal rule applies


def test_empty_inputs():
    assert combine_ppe([],[])==[]


def test_invalid_ensemble_configuration_is_rejected():
    for kwargs in ({'ensemble_ppe':True},{'vest_match_iou':0},{'vest_match_iou':1.1},{'ensemble_ppe':'yes'}):
        try:
            ModelConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'must reject {kwargs}')


if __name__=='__main__':
    failures=[]
    tests=[(n,f) for n,f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    for name,fn in tests:
        try:
            fn()
            print('PASS',name)
        except Exception as exc:
            failures.append(name)
            print('FAIL',name,repr(exc))
    print(f'{len(tests)-len(failures)}/{len(tests)} passed')
    raise SystemExit(bool(failures))
