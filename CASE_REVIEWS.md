# Five case reviews from the selected submission

All five pairs are among the highest-ranked pairs of the selected submission (A, `submission.csv`). The hand IDs are the evidence hands we
submitted for them. We do not know the evaluation labels; each review describes what is observable in the public tables and gives a plausible
innocent reading.

**Notation.** A and B are the two players of the pair: A is `player_1` and B is `player_2` in `evaluation_pairs.csv`. Other players are `o<seat>`.
Cards are written rank + suit (`Td` = ten of diamonds). "Equity" is the probability of winning at showdown among the players still in: exact
enumeration after the flop, Monte Carlo before it.

**How to reproduce.** Every hand can be printed with `case_reviews/select_cases.py`, and every count with `case_reviews/case_stats.py`. Both read
only the raw competition tables. "Folded the better hand" means that the folder's heads-up equity against the bettor was above 50% at the
moment of the fold.

---

## Case 1 — directed transfer · pair `P7B7DE55172FC` (risk rank 1 of 112,540)

**Players:** A `U3F1C42C7503C`, B `U45413518A955` · 77 shared evaluation hands · blinds 1/2
**Hands:** `H4C93A094E27395`, `HFE86FBD18E5B82`, `H9E0E7716A5BA20` (also submitted: `HBBD8584F7E5956`, `HCE649A638BA9CA`)

**Observable behaviour.**
- `H4C93A094E27395`: A (6c Tc) calls B's 3-bet. On Kh 9h Qh it calls B's pot-sized bet with 5% equity, then calls the turn bet (9%). On the river (9d) it calls B's all-in with 0% equity; it holds nothing beyond the board's pair of nines. A loses its whole 234-chip stack to B.
- `HFE86FBD18E5B82`: A flops two pair (Ks 2s on 2d Th Kd, 88% against B's pair of tens), checks, and folds to B's 11-chip bet into an 11-chip pot.
- `H9E0E7716A5BA20`: A puts 45 chips in, turns two pair (8d 2d on 8s Qc 9d 2s, 73% against B's pocket tens), checks, and folds to B's 113-chip bet.

Over the 77 shared hands, A folded to B's bets 10 times and held the better hand in 5 of them. Against other players' bets A folded 23 times and held the better hand only 4 times. Net result over the shared hands: A −1,109 chips, B +626.

**Plausible benign explanation.** A may be a weak, emotional player: calling stations do chase gutshots and make hopeless "hero" calls, and the same player can over-fold two pair on connected or two-tone boards out of fear of straights and flushes. B is the aggressor in most of their pots (47 bets or raises with A still in the hand). A losing player bleeding chips to the most aggressive player at the table is ordinary, and 10 folds is a small sample.

---

## Case 2 — directed transfer · pair `P798B26F7BE4D` (risk rank 2)

**Players:** A `U7801FE57DB5F`, B `UC75ED366AEB4` · 148 shared evaluation hands · blinds 2/4
**Hands:** `H74BA122D83A169`, `H99FB8B3384DC62`, `HBCD55C8D0B821A` (also submitted: `HE977114D03AE94`, `HE3883412F8265C`)

**Observable behaviour.** The same script repeats: A invests pre-flop, flops the best hand against B, and gives the pot up to B's bet.
- `H74BA122D83A169`: A (7h Kh) raises B's limp to 12. It flops top pair on 3h Kc 6c (79% against B's 6s 8c) and folds to B's 43-chip overbet.
- `H99FB8B3384DC62`: A (5d 8d) calls B's raise. It flops a pair of fives on 2h 9d 5h (78% against B's ace-high) and folds to a pot-sized bet.
- `HBCD55C8D0B821A`: A calls a 3-bet from B with 3c 2d. It flops a pair on As 3s Qc (74% against B's ten-high), checks, and folds to B's 116-chip bet.

Over 148 shared hands, A folded to B 20 times and held the better hand in 11 (55%). Against other bettors A folded 42 times and held the better hand in 5 (12%). B folded to A 10 times, only once with the better hand. Net over the shared hands: A −8,626 chips, B +5,528.

**Plausible benign explanation.** A could be a risk-averse "fit-or-fold" player facing a hyper-aggressive one: B made 105 bets or raises with A in the hand, and its bets are often 1–2× the pot. Against large bets, folding one pair (especially a pair of threes on an ace-high board) is defensible, and a player who overestimates aggressive ranges folds the best hand often. Entering pots with weak hands and then giving up is a common leak, not evidence of intent.

---

## Case 3 — soft play · pair `P603FF8F289F0` (risk rank 13)

**Players:** A `UB415F747C71E`, B `UF1D4D685D6C0` · 99 shared evaluation hands · blinds 2/4
**Hands:** `H90F4FFD0AD3520`, `HEDDFCDDA4726AA`, `H9E7511DA97CBEF` (also submitted: `H564582E4CEB68C`, `H7F79352B2677FB`)

**Observable behaviour.** The partners avoid betting into each other, and the pot goes to whoever bets last, even without a hand.
- `H90F4FFD0AD3520`: B raises (4c 8d), A calls (9c Qd). B holds a pair of fours on 4d As 7c 3c (77–86% equity) and checks the flop and turn. On the river A bets 26 into 22 with queen-high, and B folds the winning hand.
- `HEDDFCDDA4726AA`: three-way pot. B has a pair of kings on Ks 2h Ad 7s (96–100%) and checks the flop and turn. On the river A bets 25 with ten-high, and B folds.
- `H9E7511DA97CBEF`: A (7s Ah) has a pair of sevens on 7d 6h 8d (59% against B's ace-high) and folds to B's 9-chip bet into 12.

In the 9 hands that became heads-up between them, 9 streets were checked through by both players. Both players ended the period in profit against the table (A +582, B +714). Each folded to the other rarely (6 and 8 times).

**Plausible benign explanation.** Checking medium pairs for pot control and folding them to a river bet is standard cautious play, and a river bet into two checks is exactly the line that represents strength. Two solid players who respect each other's bets will produce few big pots against each other and still win from weaker opponents. Nine heads-up hands is also a very small sample.

---

## Case 4 — soft play · pair `P8035E280D579` (risk rank 19)

**Players:** A `U84C5E9ADC083`, B `UE8FF593D0206` · 59 shared evaluation hands · blinds 2/4
**Hands:** `H37C8A69C2270CD`, `HEFE750668E81CD`, `HD8DD0922AE1F44` (also submitted: `H489554813920ED`, `HA9B721174DE8DF`)

**Observable behaviour.**
- `H37C8A69C2270CD`: A checks the big blind with Th 9c and flops the nut straight on 8s Js 7c. Heads-up against B, A and B then check the flop, turn and river, so the pot never grows beyond 10 chips.
- `HEFE750668E81CD`: three-way, B has top pair (Kh 7h on Kc 6s 6c, 88%) and A has pocket nines. Nobody bets on any street, and B wins 29 chips at showdown.
- `HD8DD0922AE1F44`: B (Qc Ad) flops a pair of aces on Kh As Td and calls A's small bet. On the turn (93% equity) B folds to A's 28-chip bet. A holds jack-high with a gutshot and later bluffs the third player out of a 162-chip pot.

A folded to B 7 times (twice with the better hand); B folded to A 4 times (once).

**Plausible benign explanation.** Slow-playing the nuts in a tiny limped pot hoping the opponent bluffs is a recognised, if weak, tactic. Checking top pair on a paired board in a multiway pot is common pot control. Folding top pair with a queen kicker to a player who bets both the flop and the turn on A-K-T-8, where two pair, sets and a straight are all possible, is conservative but reasonable. Only 3 hands became heads-up between these two players, and both lost chips over the period (A −712, B −397), which does not fit a pair farming the table.

---

## Case 5 — coordinated isolation · pair `P751797DE1979` (risk rank 36)

**Players:** A `U24EBD56B4CC5`, B `UF44FDB443845` · 73 shared evaluation hands · blinds 2/4
**Hands:** `H16C14C6C20E05C`, `H6193C8698E5FE6`, `HE52E1B6465DC1F` (also submitted: `HACCF5957F4CFCA`, `HD990A4F5D21920`)

**Observable behaviour.** B opens first to act with weak hands. When a third player enters, the two partners re-raise around that player, then one of them steps aside for the other.
- `H16C14C6C20E05C`: B opens with 7d 4c. o3 (As Qh) calls, A (Th Ts) 3-bets, and B 4-bets to 72 with seven-four offsuit, which forces o3 out. B then folds to A's 5-bet to 252, and A collects the pot.
- `H6193C8698E5FE6`: B opens to 10 with Ac 9d. A 3-bets from the small blind with Qh 5s, and B folds ace-nine without a fight.
- `HE52E1B6465DC1F`: B opens with 5s 8s and A defends the big blind. A flops the better hand (Qc 9s on 2h Td 3d, 76%), checks, and folds to B's 15-chip bet.

B folded to A's bets or raises 21 times, the same number as to all other players combined (18). A raised in 6 pre-flop raise wars (three or more raises) and B in 4 during the 73 shared hands.

**Plausible benign explanation.** Loose-aggressive players open weak hands and 4-bet as bluffs, and a 4-bet bluff that folds to a 5-bet is the normal way such a bluff ends. A 3-bet with Q5o from the small blind against a wide opener is an aggressive but known exploit, and folding A9o to it is ordinary. Two aggressive regulars at one table will often raise in the same pots, and 73 hands cannot separate style from coordination.
