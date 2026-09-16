# A published byte count was measuring my filesystem

Status: accepted, 2026-09-16.

## What the completion pass found

The README said `configs/candidate-inverted.yml` "puts 204 bytes of RuntimeWarning on
stderr and exits 0". Every published figure was re-measured before this project was
called finished. This one came back 159.

A count that does not reproduce is broken, so it got chased rather than softened.

## The cause

A Python warning prints the absolute path of the file that raised it. The repo lives at a
different path on every machine it runs on, so the byte count moves with the directory
name. Measured at three path lengths:

    path length 20    159 bytes      /tmp/run0916/w1/repo
    path length 47    186 bytes      a deliberately long copy
    path length 68    207 bytes      the checkout that re-measured it

139 plus the length of the path, exactly, in all three. The published 204 corresponds to a
checkout path of 65 characters, which is what the machine that wrote the sentence
happened to give it.

## Why this one is worth writing down

Every other wrong number this program has caught was a wrong measurement. This was a
correct measurement of the wrong thing. Nobody mistyped it, and re-running the command on
the machine that produced it would have confirmed it.

It is also invisible to the checks. It has the shape of a fact about the model, it sits in
a paragraph about a model, and a reader who cloned the repo and got 212 would assume they
had done something wrong rather than that the number was never portable.

The general form: a figure is only publishable if it is a property of the thing being
described. This one was a property of where the thing was sitting.

## The decision

The byte count is gone. The paragraph now says what is invariant. One `RuntimeWarning`,
the text `overflow encountered in matmul`, exit code 0. Those hold on any machine.

The replaced sentence is kept in the README with the reason, rather than quietly corrected,
because the failure is more useful than the fact was.

## What this does not fix

Nothing checks that a published figure is path independent, or environment independent more
generally. This was caught by a human re-measurement on the last day of a project, which is
late and does not scale. A figure derived from a timing, a hostname, a process id or a
temporary directory would all fail the same way and none of them would be noticed.

`scripts/` has probes that print figures and the README quotes them. A check that runs a
probe from two different directories and diffs the output would have caught this one, and
it would not catch a timing. Not built.
