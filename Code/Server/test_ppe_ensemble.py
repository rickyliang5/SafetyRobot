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



def test_any_two_of_three_models_can_confirm_a_vest():
    for primary,reference,additional in (
        ([d('safety-vest')],[],[d('safety-vest')]),
        ([],[d('safety-vest')],[d('safety-vest')]),
        ([d('safety-vest')],[d('safety-vest')],[]),
    ):
        out=combine_ppe(primary,reference,additional=additional)
        assert len(out)==1 and out[0].name=='safety-vest'


def test_third_model_alone_is_not_consensus():
    out=combine_ppe([],[],additional=[d('safety-vest'),d('safety-vest')])
    assert len(out)==2 and all(x.name=='uncertain-vest' for x in out)


def test_three_votes_use_second_highest_score_not_sum_or_product():
    out=combine_ppe([d('safety-vest',conf=.9)],[d('safety-vest',conf=.7)],
                    additional=[d('safety-vest',conf=.3)])
    assert len(out)==1 and out[0].name=='safety-vest' and out[0].conf==.7


def test_third_model_adds_helmet_but_not_person_owners():
    person=d('person',PERSON)
    other=d('person',(0.,0.,100.,200.))
    helmet=d('helmet',(250.,70.,330.,130.))
    out=combine_ppe([], [person],additional=[other,helmet])
    assert person in out and helmet in out and other not in out


def test_third_vote_cannot_bridge_nonoverlapping_workers():
    # A big third box overlapping two different vests gets only one vote.
    a=d('safety-vest',(0.,0.,50.,100.))
    b=d('safety-vest',(50.,0.,100.,100.))
    c=d('safety-vest',(0.,0.,100.,100.))
    out=combine_ppe([a],[b],additional=[c])
    assert sum(x.name=='safety-vest' for x in out)==1
    assert sum(x.name=='uncertain-vest' for x in out)==1


def test_third_model_requires_ensemble_configuration():
    try:
        ModelConfig(construction_onnx='candidate.onnx')
    except ValueError:
        pass
    else:
        raise AssertionError('third model requires ensemble mode')


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
