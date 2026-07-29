"""
Offline tests for the creature-page parser (no network): wikitext infobox
extraction, normalization honesty (unknowns stay None), Special:Export XML
parsing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrape_creatures import (split_template_fields, normalize,
                              pages_from_xml, _to_int, _to_bool)

SAMPLE = """
{{CreatureInfobox
| name = Jade Oni
| school = [[Life School|Life]]
| classification = Oni
| rank = 6
| health = 4,000
| stunable = No
| beguile =
| spells = [[Centaur]], [[Seraph]] {{Pip}} cost varies
| cheats = When you place a [[Feint|trap {{tt|like this}}]], he attacks
}}
Jade Oni is the final boss of [[MooShu]].
"""


def test_infobox_fields_survive_nested_markup():
    f = split_template_fields(SAMPLE)
    assert f["name"] == "Jade Oni"
    assert f["school"] == "Life"                  # [[a|b]] -> b
    assert f["rank"] == "6"
    assert f["health"] == "4,000"
    # pipes inside nested {{ }} / [[ ]] must not split fields
    assert "cost varies" in f["spells"]
    assert "he attacks" in f["cheats"]


def test_normalize_is_honest_about_unknowns():
    rec = normalize("Jade Oni", split_template_fields(SAMPLE))
    assert rec["school"] == "life"
    assert rec["health"] == 4000
    assert rec["rank"] == 6
    assert rec["stunable"] is False
    assert rec["beguilable"] is None              # empty stays unspecified
    assert rec["confidence"] == "wiki-scrape"


def test_scalar_coercers():
    assert _to_int("12,345 HP") == 12345
    assert _to_int(None) is None
    assert _to_bool("Yes") is True and _to_bool("no") is False
    assert _to_bool("sometimes") is None


def test_no_infobox_returns_empty():
    assert split_template_fields("just prose, no template") == {}


def test_special_export_xml(tmp_path):
    xml = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
      <page><title>Rattlebones</title>
        <revision><text>{{CreatureInfobox
| school = Death
| health = 500
| stunable = Yes
}}</text></revision></page>
      <page><title>Empty Page</title>
        <revision><text></text></revision></page>
    </mediawiki>"""
    p = tmp_path / "export.xml"
    p.write_text(xml)
    pages = dict(pages_from_xml(str(p)))
    assert "Rattlebones" in pages
    rec = normalize("Rattlebones",
                    split_template_fields(pages["Rattlebones"]))
    assert rec == dict(name="Rattlebones", school="death", rank=None,
                       health=500, stunable=True, beguilable=None,
                       classification=None, confidence="wiki-scrape",
                       fields={"school": "Death", "health": "500",
                               "stunable": "Yes"})
