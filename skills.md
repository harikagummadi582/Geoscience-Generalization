# AI-assisted coding: practice checklist

Come back to this file to check that I'm working on this project the way a software engineer is
expected to work with AI. Formats vary by company, but AI-assisted coding interviews tend to check
the same things, so I practice them here for real.

## What AI-assisted coding interviews tend to check

- **Planning before prompting:** state the goal, constraints and a rough plan first, instead of
  pasting the problem and accepting whatever comes back.
- **Small, specific prompts with context:** one change at a time, with the relevant files and the
  expected output named.
- **Reading and verifying the AI's output:** catch bugs, question assumptions and run tests rather
  than trusting it.
- **Debugging and judgment:** when the AI is wrong, say why and steer it, or fix it myself.
- **Communication:** narrate my reasoning out loud as I work.

## How I practice on this project

1. **Plan first.** Before each task, write a 3-line plan (goal, inputs, how I'll check it) in
   `NOTES.md` or the PR description, then prompt from it.
2. **Review every AI change.** Read `git diff` before committing, and write my own commit messages.
3. **Keep an AI log.** Note where the AI was wrong or misleading (see below). Real examples make
   strong interview stories.
4. **Write my own checks.** Tests or sanity checks, e.g. "windows never span depth gaps".
5. **Sometimes work without AI.** Do a task with no AI and explain the approach out loud, since
   some rounds ban or limit it.

## Per-task checklist (copy into NOTES.md or the PR)

- [ ] Goal, inputs and how I'll check it written down (3 lines)
- [ ] Prompt names the files, the change, and the expected output
- [ ] I read the whole diff and can explain every line
- [ ] I ran it and checked the result against something independent
- [ ] I wrote or ran at least one sanity check myself
- [ ] My own commit message; PR description says what changed and what is still uncertain
- [ ] Added anything surprising to the AI log

## AI log (where the AI was wrong or misleading)

| Date | What the AI said or did | What was actually true | How I caught it |
|---|---|---|---|
| 2026-10-06 | Warned the Eaton formula could leak the target into the features, based on the synthetic generator | On the real data, the n=3 Eaton formula does not reproduce PPP (R² very negative), so that leakage claim was wrong | Checked the claim against the real CSVs instead of trusting the generator code |
| 2026-10-06 | Called the proposal still due and the Oct 12 report "4" | Proposal was already submitted; Oct 12 is Report-3 | Compared against my Canvas grades |
| 2026-10-06 | Profiled the data and found 0 missing values | Missing values are encoded as -999.25 (or -999), not NaN | Looked at min values and noticed the sentinel |
| 2026-10-06 | Treated PPP as a measured pressure | The data README says PPP is a predicted curve, so results are agreement with PPP, not error vs measurement | Read the data README before reporting |

## Sanity checks to write for this project

- Windows never span a depth gap
- No well appears in both train and test (split by well, never by row)
- Every well is resampled to about the same spacing before pooling (PINDORI-1/2 are ~4x finer)
- Merged window predictions cover every sample of the held-out well exactly once after averaging
