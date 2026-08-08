# Preset scripts

Drop deimoslang quester scripts here (`*.txt` or `*.deimos`) and they
appear in the script dialog's preset list instead of having to be pasted
every session.

Two things happen to a preset that do not happen to pasted text:

- **The party's names are filled in.** `Main_Account`, `Questee2..4` and
  their `_School` partners are set from the wizards actually hooked, in
  seat order — see `scripts.configure`. Only values still at the
  script's own placeholder are touched; anything you typed is yours.
  This is deliberate: an unconfigured quester does not fail, it silently
  skips every friend-teleport it has, and a party that cannot regroup
  stalls. One run lost forty minutes to exactly that.

- **The title comes from the file.** Add `# @name: Something Readable`
  near the top, or the filename is used.

These are other people's scripts. Keep whatever authorship and licence
header the file came with — wizAi does not claim them, and the ones in
circulation (TTS Arc 1 and its relatives) are community work.
