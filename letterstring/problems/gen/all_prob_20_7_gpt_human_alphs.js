var all_problems = {
  "alph_0": {
    "shuffled_letters": [
      "u",
      "n",
      "o",
      "y",
      "q",
      "l",
      "f",
      "j",
      "d",
      "h",
      "a",
      "z",
      "b",
      "e",
      "w",
      "m",
      "c",
      "i",
      "x",
      "v"
    ],
    "shuffled_alphabet": [
      "u",
      "n",
      "o",
      "y",
      "q",
      "l",
      "g",
      "f",
      "j",
      "d",
      "k",
      "h",
      "a",
      "z",
      "b",
      "p",
      "e",
      "r",
      "s",
      "t",
      "w",
      "m",
      "c",
      "i",
      "x",
      "v"
    ],
    "larger_int": [
      {
        "prompt": "[g f j d] &nbsp [l f j d]<br>[g j k a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[k j a z b] &nbsp [k h a z b]<br>[v f d h z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[o y l q g] &nbsp [o y q l g]<br>[g j b a k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[h b z a p] &nbsp [h a z b p]<br>[n d l f y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[q l g f] &nbsp [q l g j]<br>[s w c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[y n o u q] &nbsp [u n o y q]<br>[z r p t m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[a z b p] &nbsp [a z b e]<br>[k a b e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[p e r s] &nbsp [p e r t]<br>[l f d h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[n o y q] &nbsp [u o y q]<br>[l f d h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[e r s t] &nbsp [e r s t w]<br>[f d h z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "longer_targ": [
      {
        "prompt": "[g f j d k k] &nbsp [g f j d k]<br>[u n o y q q l g f j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[y q l g] &nbsp [y q l f]<br>[d k h a z b p e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[k h h a z b] &nbsp [k h a z b]<br>[g f j d k h a z b b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[u n o y y q] &nbsp [u n o y q]<br>[a a z b p e r s t w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[u n o y q q] &nbsp [u n o y q]<br>[y q l g f f j d k h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[o y q l] &nbsp [o y q g]<br>[u n o y q l g f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[j d k h] &nbsp [f d k h]<br>[q l g f j d k h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[y q l g] &nbsp [y q l f]<br>[r s t w m c i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[j d k h] &nbsp [j d k h a]<br>[d k h a z b p e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[e r s t] &nbsp [e r s w]<br>[r s t w m c i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "group": [
      {
        "prompt": "[c i x v] &nbsp [m i x v]<br>[l l g g f f j j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[d k h a] &nbsp [d k h a z]<br>[m m c c i i x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[k h a z] &nbsp [k h a b]<br>[f f j j d d k k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[j d k h] &nbsp [j d k a]<br>[d d k k h h a a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[c a z b p] &nbsp [h a z b p]<br>[u u n n o o t t q q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[i m c w x] &nbsp [w m c i x]<br>[o o y y g g l l q q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[m c i x] &nbsp [m c i v]<br>[h h a a z z b b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[e r r s t w] &nbsp [e r s t w]<br>[u u n n n n o o y y q q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[m c i x] &nbsp [m c i v]<br>[s s t t w w m m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[c i x v] &nbsp [m i x v]<br>[m m c c i i x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "interleaved": [
      {
        "prompt": "[p n o y q] &nbsp [u n o y q]<br>[b x p x e x m x s x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[f n d k h] &nbsp [f j d k h]<br>[q x l x r x f x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[q w g f j] &nbsp [q l g f j]<br>[a x z x n x p x e x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[e r s t w w] &nbsp [e r s t w]<br>[s x t x w x m x c x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[m c i x] &nbsp [m c i x v]<br>[a x z x b x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[l g f j] &nbsp [l g f j d]<br>[m x c x i x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[e b p z r] &nbsp [z b p e r]<br>[s x w x t x m x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[x c i m v] &nbsp [m c i x v]<br>[m x w x c x i x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[n o o y q l] &nbsp [n o y q l]<br>[s x t x w x m x c x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[m c i x] &nbsp [m c i v]<br>[k x h x a x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "letter2num": [
      {
        "prompt": "[q l g f] &nbsp [q l g j]<br>[20 21 22 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[k h a z b b] &nbsp [k h a z b]<br>[9 10 11 12 13 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[e r s t] &nbsp [e r s t w]<br>[22 23 24 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[l g f j j d] &nbsp [l g f j d]<br>[17 18 19 20 21 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[b p e r s s] &nbsp [b p e r s]<br>[7 8 8 9 10 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[x t w m c] &nbsp [s t w m c]<br>[12 10 14 15 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[d k h a] &nbsp [d k h z]<br>[11 12 13 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[h a d b p] &nbsp [h a z b p]<br>[1 2 3 25 5] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[c i x v] &nbsp [m i x v]<br>[13 14 15 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[l q y g f] &nbsp [y q l g f]<br>[14 11 12 13 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "reverse": [
      {
        "prompt": "[w m c i x x] &nbsp [w m c i x]<br>[d j f g g l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[j d k h a a] &nbsp [j d k h a]<br>[v x i c c m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[u n o o y q] &nbsp [u n o y q]<br>[l q y o o n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[l g f j] &nbsp [l g f d]<br>[r e p b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[m c i x] &nbsp [m c i v]<br>[d j f g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[h a z b] &nbsp [h a z p]<br>[e p b z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[m c i x] &nbsp [m c i v]<br>[w t s r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[k k h a z b] &nbsp [k h a z b]<br>[f f g l q y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[n o y q] &nbsp [n o y q l]<br>[h k d j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[k h a z z b] &nbsp [k h a z b]<br>[w w t s r e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split1": [
      {
        "prompt": "[p a z b h] &nbsp [h a z b p]<br>[u u q q o o g g j j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[b p e r] &nbsp [b p e s]<br>[m x w x t x s x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[u n o y] &nbsp [u n o q]<br>[r r s s t t w w m m c c i i x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[u n o i q] &nbsp [u n o y q]<br>[24 23 22 21 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[d k h a] &nbsp [d k h z]<br>[j f g l q y o n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[t w m c] &nbsp [t w m c i]<br>[q q y y o o n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[t s r w m] &nbsp [r s t w m]<br>[1 2 3 7 5 6 4 8 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[a z b p e e] &nbsp [a z b p e]<br>[11 10 9 8 7 7] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[w m c i] &nbsp [w m c i x]<br>[u x n x o x y x q x l x g x f x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[g f d j k] &nbsp [g f j d k]<br>[l x g x f x d x j x k x h x a x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split2": [
      {
        "prompt": "[k h l z b] &nbsp [k h a z b]<br>[r x t x m x i x f x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[y q l z f] &nbsp [y q l g f]<br>[16 15 17 13 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[u n o y] &nbsp [u n o y q]<br>[17 17 18 18 19 19 20 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[f j d k] &nbsp [f j d h]<br>[y x l x f x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[l g f d j] &nbsp [l g f j d]<br>[2 4 8 6 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[b g f j d] &nbsp [l g f j d]<br>[7 20 9 10 11 12 13 14 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[a z b p] &nbsp [a z b p e]<br>[q q l l g g f f j j d d k k h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[q l g f] &nbsp [y l g f]<br>[x x i x c x m x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[g f j m k] &nbsp [g f j d k]<br>[f f j j d d k k h h n n z z b b p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[w m c i] &nbsp [w m c x]<br>[7 7 8 8 9 9 10 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split3": [
      {
        "prompt": "[y q l g] &nbsp [y q l g f]<br>[b x p x e x r x s x t x w x m x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[a z b p] &nbsp [a z b e]<br>[f f d d h h z z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[l g j f d] &nbsp [l g f j d]<br>[7 0 8 0 11 0 10 0 9 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[d k h a] &nbsp [d k h z]<br>[6 6 7 7 8 8 9 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[y g l q f] &nbsp [y q l g f]<br>[9 9 6 6 7 7 8 8 5 5] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[f q l g y] &nbsp [y q l g f]<br>[g x y x q x l x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[e b p z r] &nbsp [z b p e r]<br>[s x m x w x t x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[p e r s] &nbsp [p e r t]<br>[b b e e s s w w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[r s t w] &nbsp [e s t w]<br>[q q g g j j k k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[s t w m] &nbsp [r t w m]<br>[p r t m i v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split4": [
      {
        "prompt": "[h a z b] &nbsp [h a z p]<br>[d d x x k k x x h h x x a a x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[a z b p] &nbsp [h z b p]<br>[17 18 19 20 21 22 23 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[j d k h] &nbsp [f d k h]<br>[f g l q y o n u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[y q l g] &nbsp [o q l g]<br>[18 19 20 21 22 23 24 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[z b p e] &nbsp [z b p r]<br>[12 11 10 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[a z b p] &nbsp [a z b e]<br>[w w x x m m x x c c x x i i x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[p b z e r] &nbsp [z b p e r]<br>[i i x x w w x x m m x x c c x x t t x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[g y q l o] &nbsp [o y q l g]<br>[f x d x h x p x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[f o y q l] &nbsp [n o y q l]<br>[g g k k d d j j f f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[r m t w s] &nbsp [r s t w m]<br>[j j x x g g x x f f x x l l x x d d x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split5": [
      {
        "prompt": "[s t w m j] &nbsp [s t w m c]<br>[a a h h k k d d z z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[m c i i x v] &nbsp [m c i x v]<br>[h h k d j f g l q y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[e r s t] &nbsp [e r s w]<br>[14 13 12 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[t w m c c i] &nbsp [t w m c i]<br>[14 16 18 18 20 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[d k h a] &nbsp [d k h a z]<br>[3 4 5 6 7 8 9 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[l o y q n] &nbsp [n o y q l]<br>[l f d h z r p t m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[l g f j] &nbsp [q g f j]<br>[n x o x y x q x l x g x f x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[k d h a z] &nbsp [d k h a z]<br>[2 4 3 5 1] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[r f j d k] &nbsp [g f j d k]<br>[g g l l q q m m o o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[z b p e] &nbsp [z b p r]<br>[5 7 9 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split6": [
      {
        "prompt": "[t c m w i] &nbsp [t w m c i]<br>[4 10 8 6 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[u n o y] &nbsp [u n o q]<br>[l l f f d d h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[g f j d] &nbsp [g f j k]<br>[18 19 20 21 22 23 24 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[k a h z b] &nbsp [k h a z b]<br>[l z d h f p r t m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[p e r s] &nbsp [b e r s]<br>[14 14 15 15 16 16 17 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[w m t i x] &nbsp [w m c i x]<br>[k k s s t t w w m m c c i i x x v v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[s t p m c] &nbsp [s t w m c]<br>[z x b x p x e x r x s x t x v x m x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[j g f l d] &nbsp [l g f j d]<br>[14 14 13 13 15 15 16 16 17 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[t w m c] &nbsp [t w m i]<br>[j j x x d d x x k k x x h h x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[y q l g] &nbsp [y q l g f]<br>[j g q o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gensplit7": [
      {
        "prompt": "[m c i x] &nbsp [m c i v]<br>[2 4 6 8] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[m c i x] &nbsp [m c i v]<br>[12 0 13 0 14 0 15 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[u n o y] &nbsp [u n o q]<br>[q q x x l l x x g g x x f f x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[a z r p e] &nbsp [a z b p e]<br>[q q y y o o n n z z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[n o y q] &nbsp [u o y q]<br>[j j d d k k h h a a z z b b p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[m m c i x v] &nbsp [m c i x v]<br>[7 7 8 9 10 11 12 13 14 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[l g f j] &nbsp [q g f j]<br>[e e s s w w c c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[w m c x i] &nbsp [w m c i x]<br>[g g f f j j d d k k z z a a h h b b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[m c c i x v] &nbsp [m c i x v]<br>[8 10 12 14 14 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[k h a z] &nbsp [k h a b]<br>[b x p x e x r x s x t x w x m x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split1": [
      {
        "prompt": "[o y q l] &nbsp [o y q g]<br>[25 0 24 0 23 0 22 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[g f j d] &nbsp [l f j d]<br>[p p x x b b x x z z x x a a x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[q l g f] &nbsp [y l g f]<br>[f f l l y y n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[u n o y v] &nbsp [u n o y q]<br>[v v x x s s x x r r x x e e x x p p x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[w m c i] &nbsp [w m c x]<br>[l x f x d x h x z x p x r x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[b p e r] &nbsp [b p e s]<br>[j j f f g g l l q q y y o o n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[c i x v] &nbsp [m i x v]<br>[5 5 4 4 3 3 2 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[a z b p] &nbsp [a z b e]<br>[c w s e b a k j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[a z b p] &nbsp [a z b e]<br>[18 18 19 19 20 20 21 21 22 22 23 23 24 24 25 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[e r s t] &nbsp [e r s t w]<br>[3 3 5 5 7 7 9 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split2": [
      {
        "prompt": "[n o y q] &nbsp [n o y q l]<br>[b b x x p p x x e e x x r r x x s s x x t t x x w w x x m m x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[o j q l g] &nbsp [o y q l g]<br>[x x x x t t x x s s x x r r x x e e x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[d k h a] &nbsp [d k h a z]<br>[2 2 0 0 3 3 0 0 4 4 0 0 5 5 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[m c i x] &nbsp [m c i v]<br>[17 16 15 14 13 12 11 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[d k h a] &nbsp [d k h a z]<br>[g g x x j j x x k k x x a a x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[m w t c i] &nbsp [t w m c i]<br>[f f d d p p z z h h r r t t m m i i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[g f j d] &nbsp [g f j k]<br>[x c w s e b a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[g l f j d] &nbsp [l g f j d]<br>[w w x x s s x x e e x x c c x x x x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[u n o y] &nbsp [u n o y q]<br>[n n x x y y x x l l x x f f x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[l d f j g] &nbsp [l g f j d]<br>[z x l x f x d x h x y x p x r x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split3": [
      {
        "prompt": "[n o c q l] &nbsp [n o y q l]<br>[1 3 5 26 9 11 13 15 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[o y q l] &nbsp [n y q l]<br>[10 8 6 4] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[u n o y] &nbsp [u n o y q]<br>[10 10 12 12 14 14 16 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[n o y q] &nbsp [n o y l]<br>[14 0 15 0 16 0 17 0 18 0 19 0 20 0 21 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[h a z b] &nbsp [h a z b p]<br>[d d j j f f g g l l q q y y o o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[t w m c v] &nbsp [t w m c i]<br>[b x z x a x o x k x d x j x f x g x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[s t w m] &nbsp [s t w m c]<br>[24 0 23 0 22 0 21 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[o l q y g] &nbsp [o y q l g]<br>[22 0 21 0 18 0 19 0 20 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[n o y q] &nbsp [u o y q]<br>[13 15 17 19 21 23 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[s t w m m c] &nbsp [s t w m c]<br>[v x v x i x m x t x r x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split4": [
      {
        "prompt": "[l g f j] &nbsp [l g f d]<br>[9 8 7 6 5 4 3 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[j d k h] &nbsp [j d k h a]<br>[6 6 8 8 10 10 12 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[z b p e] &nbsp [z b p e r]<br>[6 6 8 8 10 10 12 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[o y q l] &nbsp [o y q g]<br>[15 13 11 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[t w m c] &nbsp [t w m i]<br>[t x r x p x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[d k h a a z] &nbsp [d k h a z]<br>[n x y x l x f x d x h x z x z x p x r x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[p w m c i] &nbsp [t w m c i]<br>[9 11 13 7 17 19 21 23 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[n o y q] &nbsp [n o y l]<br>[r x p x z x h x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[m c d x v] &nbsp [m c i x v]<br>[12 12 11 11 10 10 3 3 8 8] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[w m c x i] &nbsp [w m c i x]<br>[z i m t r p v h d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split5": [
      {
        "prompt": "[q n o y u] &nbsp [u n o y q]<br>[7 0 8 0 9 0 13 0 11 0 12 0 10 0 14 0 15 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[m c i h v] &nbsp [m c i x v]<br>[20 20 15 15 17 17 19 19 21 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[m c i x] &nbsp [m c i x v]<br>[h h x x k k x x d d x x j j x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[e r s t] &nbsp [e r s t w]<br>[2 4 6 8 10 12 14 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[p a z b h] &nbsp [h a z b p]<br>[k k x x a a x x h h x x z z x x d d x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[o y y q l g] &nbsp [o y q l g]<br>[g g j j k k a a b b b b e e s s w w c c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[o s q l g] &nbsp [o y q l g]<br>[v v x x i i c c n n w w t t s s r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[d k h a] &nbsp [j k h a]<br>[13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[t w m c] &nbsp [s w m c]<br>[x x x x i i x x c c x x m m x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[w m c i] &nbsp [w m c x]<br>[t t r r p p z z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split6": [
      {
        "prompt": "[k h a z] &nbsp [k h a b]<br>[t x r x p x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[t n m c i] &nbsp [t w m c i]<br>[k x l x f x d x h x z x p x r x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[s t w m] &nbsp [s t w m c]<br>[17 17 19 19 21 21 23 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[g f j d] &nbsp [g f j k]<br>[j x g x q x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[t w w m c i] &nbsp [t w m c i]<br>[y x l x f x d x h x z x z x p x r x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[n o y q] &nbsp [n o y q l]<br>[13 0 15 0 17 0 19 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[m c i x u] &nbsp [m c i x v]<br>[m m a a k k j j g g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[b p e r] &nbsp [z p e r]<br>[17 0 19 0 21 0 23 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[d k h a] &nbsp [d k h z]<br>[5 5 7 7 9 9 11 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[p e r s s t] &nbsp [p e r s t]<br>[n n n n y y l l f f d d h h z z p p r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gensplit7": [
      {
        "prompt": "[u n n o y q] &nbsp [u n o y q]<br>[19 17 17 15 13 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[q l g f] &nbsp [q l g j]<br>[n x y x l x f x d x h x z x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[l d f j g] &nbsp [l g f j d]<br>[p p s s r r e e t t b b z z a a h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[b r e p s] &nbsp [b p e r s]<br>[18 18 0 0 19 19 0 0 20 20 0 0 22 22 0 0 21 21 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[r k h a z] &nbsp [d k h a z]<br>[6 6 7 7 8 8 9 9 10 10 3 3 12 12 13 13 14 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[j d k h] &nbsp [j d k a]<br>[p x z x h x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[p e r d t] &nbsp [p e r s t]<br>[j x f x g x l x q x s x o x n x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[l g m j d] &nbsp [l g f j d]<br>[19 18 17 16 23 14 13 12 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[u n y o q] &nbsp [u n o y q]<br>[23 23 19 19 20 20 21 21 22 22 18 18 24 24 25 25 26 26] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[u u n o y q] &nbsp [u n o y q]<br>[e e x x s s x x s s x x w w x x c c x x x x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ]
  },
  "alph_1": {
    "shuffled_letters": [
      "g",
      "z",
      "y",
      "w",
      "m",
      "r",
      "d",
      "t",
      "s",
      "j",
      "e",
      "o",
      "f",
      "l",
      "u",
      "v",
      "p",
      "n",
      "q",
      "c"
    ],
    "shuffled_alphabet": [
      "a",
      "b",
      "g",
      "z",
      "y",
      "w",
      "m",
      "h",
      "i",
      "r",
      "k",
      "d",
      "t",
      "s",
      "j",
      "e",
      "o",
      "f",
      "l",
      "u",
      "v",
      "p",
      "n",
      "x",
      "q",
      "c"
    ],
    "larger_int": [
      {
        "prompt": "[i h m r k] &nbsp [m h i r k]<br>[j l o v n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[m h i r] &nbsp [m h i r k]<br>[t j o l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[s j e e o f] &nbsp [s j e o f]<br>[g y m i i k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[z y w m] &nbsp [z y w m h]<br>[m i k t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[g m y w z] &nbsp [g z y w m]<br>[j i k t m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[n x q c] &nbsp [p x q c]<br>[w h r d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[k d t s] &nbsp [k d t j]<br>[j o l v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[b b g z y w] &nbsp [b g z y w]<br>[t j j o l v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[z y w m] &nbsp [z y w m h]<br>[y m i k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[w m h i] &nbsp [y m h i]<br>[j o l v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "longer_targ": [
      {
        "prompt": "[d t s e j] &nbsp [d t s j e]<br>[f l u v p n q x c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[o f l u] &nbsp [o f l v]<br>[e o f l u v p n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[v u p n x] &nbsp [u v p n x]<br>[a b g z y w i h m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[f l u v v p] &nbsp [f l u v p]<br>[y w m h i r r k d t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[y w m h] &nbsp [y w m i]<br>[f l u v p n x q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[r k d t] &nbsp [r k d s]<br>[s j e o f l u v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[g z y w] &nbsp [b z y w]<br>[d t s j e o f l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[a z g b y] &nbsp [a b g z y]<br>[v l u f p n x q c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[e o f l] &nbsp [e o f u]<br>[f l u v p n x q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[z y h m w] &nbsp [z y w m h]<br>[d t s j e o f u l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "group": [
      {
        "prompt": "[l u v p] &nbsp [l u v n]<br>[m m h h i i r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[z y w m] &nbsp [z y w m h]<br>[p p n n x x q q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[p n x q] &nbsp [p n x c]<br>[l l u u v v p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[n x q c] &nbsp [p x q c]<br>[u u v v p p n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[s j e o] &nbsp [s j e o f]<br>[t t s s j j e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[v p n x] &nbsp [v p n q]<br>[m m h h i i r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[f c u v p] &nbsp [f l u v p]<br>[j j z z o o f f l l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[f l l u v p] &nbsp [f l u v p]<br>[z z y y w w m m m m h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[r k d t] &nbsp [r k d s]<br>[v v p p n n x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[d t j s e] &nbsp [d t s j e]<br>[w w y y m m h h i i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "interleaved": [
      {
        "prompt": "[s j e o] &nbsp [s j e o f]<br>[u x v x p x n x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[i r k d] &nbsp [h r k d]<br>[h x i x r x k x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[z y w m] &nbsp [z y w h]<br>[e x o x f x l x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[b g z y w w] &nbsp [b g z y w]<br>[s x j x e x e x o x f x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[e o f l] &nbsp [j o f l]<br>[w x m x h x i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[t s j e] &nbsp [t s j e o]<br>[d x t x s x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[s j o e f] &nbsp [s j e o f]<br>[d x t x s x e x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[v p n x q q] &nbsp [v p n x q]<br>[k x d x t x s x s x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[i d k r t] &nbsp [i r k d t]<br>[f x u x l x v x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[b g z y] &nbsp [b g z y w]<br>[o x f x l x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "letter2num": [
      {
        "prompt": "[v u p n x] &nbsp [u v p n x]<br>[9 10 11 13 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[l u t p n] &nbsp [l u v p n]<br>[18 19 1 21 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[d t s j] &nbsp [k t s j]<br>[14 15 16 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[y w m h] &nbsp [y w m i]<br>[11 12 13 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[l u v p n n] &nbsp [l u v p n]<br>[22 23 24 25 26 26] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[s j e o] &nbsp [s j e f]<br>[20 21 22 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[j e o f l l] &nbsp [j e o f l]<br>[7 8 9 10 11 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[j e o f] &nbsp [j e o l]<br>[8 9 10 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[v p n x] &nbsp [v p n x q]<br>[14 15 16 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[t s j e] &nbsp [t s j e o]<br>[21 22 23 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "reverse": [
      {
        "prompt": "[y w m h] &nbsp [z w m h]<br>[q x n p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[m r i h k] &nbsp [m h i r k]<br>[e o j s t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[j j e o f l] &nbsp [j e o f l]<br>[i h m w y y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[h i r k] &nbsp [h i r d]<br>[x n p v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[u v p x n] &nbsp [u v p n x]<br>[s t r k d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[t s j e] &nbsp [t s j o]<br>[m w y z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[e o f l] &nbsp [e o f l u]<br>[n p v u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[d k r t s] &nbsp [r k d t s]<br>[c n x q p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[k k d t s j] &nbsp [k d t s j]<br>[c q x n n p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[l u v b n] &nbsp [l u v p n]<br>[j s l d k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split1": [
      {
        "prompt": "[y w m h] &nbsp [y w m h i]<br>[13 15 17 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[i h m r k] &nbsp [m h i r k]<br>[x x n n v v p p u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[t s j e v] &nbsp [t s j e o]<br>[b b l l f f o o e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[n x q c] &nbsp [p x q c]<br>[20 20 21 21 22 22 23 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[s j e o] &nbsp [s j e o f]<br>[d r h w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[p n x q] &nbsp [p n x q c]<br>[y x w x m x h x i x r x k x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[p n x q] &nbsp [p n x c]<br>[i k t j o l v n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[z y w m] &nbsp [g y w m]<br>[19 19 20 20 21 21 22 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[w m r i h] &nbsp [w m h i r]<br>[9 0 10 0 11 0 13 0 12 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[f l u v] &nbsp [o l u v]<br>[22 22 23 23 24 24 25 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split2": [
      {
        "prompt": "[y w h m i] &nbsp [y w m h i]<br>[s x j x e x o x f x l x p x v x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[p n x q] &nbsp [p n x c]<br>[d r h w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[v p n x] &nbsp [v p n q]<br>[h r d s e f u p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[o f l u] &nbsp [o f l u v]<br>[12 13 14 15 16 17 18 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[k d t s] &nbsp [k d t j]<br>[20 19 18 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[u m h i r] &nbsp [w m h i r]<br>[i x w x h x r x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[m h i r] &nbsp [m h i k]<br>[f f l l u u v v p p n n x x q q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[p n x q] &nbsp [p n x c]<br>[12 12 13 13 14 14 15 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[e o f l] &nbsp [e o f u]<br>[i h m w y z g b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[v h n x q] &nbsp [v p n x q]<br>[l x j x s x t x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split3": [
      {
        "prompt": "[e o f l] &nbsp [j o f l]<br>[o o x x f f x x l l x x u u x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[r k d t] &nbsp [r k d s]<br>[q x x x n x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[b g z y] &nbsp [b g z w]<br>[a a b b g g z z y y w w m m h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[k d t s] &nbsp [k d t j]<br>[g y m i k t j o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[h i r k] &nbsp [h i r d]<br>[10 11 12 13 14 15 16 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[z y w m] &nbsp [z y w h]<br>[w h r d s e f u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[z y w m] &nbsp [z y w m h]<br>[f x o x e x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[z y w m m h] &nbsp [z y w m h]<br>[i x h x m x w x y x y x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[o f h u v] &nbsp [o f l u v]<br>[f x a x p x x x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[j e o f] &nbsp [j e o f l]<br>[z z x x y y x x w w x x m m x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split4": [
      {
        "prompt": "[u v p f x] &nbsp [u v p n x]<br>[q q r r i i h h m m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[l u g p n] &nbsp [l u v p n]<br>[r x s x e x f x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[m h i r] &nbsp [m h i k]<br>[22 0 23 0 24 0 25 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[t s j e o o] &nbsp [t s j e o]<br>[j j j j o o l l v v n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[f l u v] &nbsp [f l u v p]<br>[22 0 23 0 24 0 25 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[v p n x] &nbsp [u p n x]<br>[7 6 5 4] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[p n x q o] &nbsp [p n x q c]<br>[f o e j s q d k r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[z y w m] &nbsp [z y w h]<br>[s x e x f x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[f l u v] &nbsp [f l u v p]<br>[j s t d k r i h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[u v p n] &nbsp [l v p n]<br>[12 0 13 0 14 0 15 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split5": [
      {
        "prompt": "[v p n x] &nbsp [v p n x q]<br>[x x n x p x v x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[d t s j] &nbsp [k t s j]<br>[21 0 22 0 23 0 24 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[d k t s j] &nbsp [k d t s j]<br>[e e o o u u l l f f v v p p n n x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[k d t s] &nbsp [k d t j]<br>[h h x x i i x x r r x x k k x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[r k c t s] &nbsp [r k d t s]<br>[p x y x u x l x f x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[i m h w r] &nbsp [w m h i r]<br>[1 0 2 0 4 0 3 0 5 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[l u v p] &nbsp [l u v n]<br>[a a g g y y m m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[j j e o f l] &nbsp [j e o f l]<br>[h x m x m x w x y x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[j e o f] &nbsp [j e o f l]<br>[y m i k t j o l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[r k d t] &nbsp [r k d s]<br>[e e j j s s t t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split6": [
      {
        "prompt": "[s j e o] &nbsp [t j e o]<br>[w w m m h h i i r r k k d d t t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[c w m h i] &nbsp [y w m h i]<br>[i i o o l l v v n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[f v u l p] &nbsp [f l u v p]<br>[16 16 20 20 18 18 19 19 17 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[f e o j l] &nbsp [j e o f l]<br>[z x w x d x r x h x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[j s e o f] &nbsp [s j e o f]<br>[z z m m w w y y h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[d t s j] &nbsp [d t s e]<br>[7 9 11 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[d t s j] &nbsp [d t s j e]<br>[20 20 21 21 22 22 23 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[b g y z w] &nbsp [b g z y w]<br>[r x k x d x s x t x j x e x o x f x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[u v p n] &nbsp [u v p x]<br>[n p v u l f o e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[f l u v o] &nbsp [f l u v p]<br>[24 24 14 14 15 15 16 16 17 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gensplit7": [
      {
        "prompt": "[y w m h] &nbsp [y w m i]<br>[21 0 22 0 23 0 24 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[p n x q] &nbsp [p n x c]<br>[x n p v u l f o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[f l u v c] &nbsp [f l u v p]<br>[g x y x w x i x k x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[y w w m h i] &nbsp [y w m h i]<br>[z z w w h h r r r r d d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[u v p x n] &nbsp [u v p n x]<br>[f f l l u u v v n n p p x x q q c c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[u v p j x] &nbsp [u v p n x]<br>[15 15 3 3 17 17 18 18 19 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[s o e j f] &nbsp [s j e o f]<br>[20 0 21 0 23 0 22 0 24 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[o f l u] &nbsp [o f l v]<br>[i i x x r r x x k k x x d d x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[g z y w] &nbsp [g z y m]<br>[i x r x k x d x t x s x j x e x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[v p n x z] &nbsp [v p n x q]<br>[2 0 3 0 4 0 10 0 6 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split1": [
      {
        "prompt": "[d t s j] &nbsp [d t s e]<br>[r x d x s x e x f x u x p x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[c n x q p] &nbsp [p n x q c]<br>[q x x x n x p x v x f x l x u x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[d d t s j e] &nbsp [d t s j e]<br>[b x z x w x h x r x r x d x s x e x f x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[j e o f] &nbsp [s e o f]<br>[19 17 15 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[h i r r k d] &nbsp [h i r k d]<br>[p x u x f x e x s x s x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[l u v p] &nbsp [l u v n]<br>[n x v x l x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[e o f l] &nbsp [e o f u]<br>[20 20 0 0 21 21 0 0 22 22 0 0 23 23 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[b h z y w] &nbsp [b g z y w]<br>[23 0 4 0 3 0 2 0 1 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[r f l u v] &nbsp [o f l u v]<br>[i x k x t x j x o x l x v x n x s x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[f l u v] &nbsp [f l u p]<br>[i i k k t t j j o o l l v v n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split2": [
      {
        "prompt": "[k d t s] &nbsp [k d t s j]<br>[4 0 6 0 8 0 10 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[a b g z] &nbsp [a b g z y]<br>[11 11 0 0 12 12 0 0 13 13 0 0 14 14 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[q n x p c] &nbsp [p n x q c]<br>[17 17 18 18 19 19 24 24 21 21 22 22 23 23 20 20 25 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[o f l u u v] &nbsp [o f l u v]<br>[q q x x x x x x n n x x p p x x v v x x v v x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[s j e o d] &nbsp [s j e o f]<br>[8 8 14 14 16 16 18 18 20 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[r i k d t] &nbsp [i r k d t]<br>[15 14 9 12 11 10 13 8 7] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[r k d t] &nbsp [i k d t]<br>[l l o o j j t t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[j e o f] &nbsp [j e o f l]<br>[v x l x o x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[f v u l p] &nbsp [f l u v p]<br>[9 0 8 0 7 0 5 0 6 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[t s j e e o] &nbsp [t s j e o]<br>[d x r x r x h x w x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split3": [
      {
        "prompt": "[k d t s] &nbsp [k d t s j]<br>[r r x x i i x x h h x x m m x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[f l u v] &nbsp [o l u v]<br>[21 21 20 20 19 19 18 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[j e o f] &nbsp [j e o f l]<br>[p u f e s d r h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[d k t s j] &nbsp [k d t s j]<br>[w w e e r r d d s s h h f f u u p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[u v p j x] &nbsp [u v p n x]<br>[21 0 20 0 12 0 18 0 17 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[p n x c q] &nbsp [p n x q c]<br>[s u f e p d r h w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[p n x q] &nbsp [p n x c]<br>[c c x x p p u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[d t s j] &nbsp [d t s e]<br>[y y m m i i k k t t j j o o l l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[o f l u] &nbsp [e f l u]<br>[w w h h r r d d s s e e f f u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[y i m h w] &nbsp [y w m h i]<br>[15 0 17 0 16 0 18 0 14 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split4": [
      {
        "prompt": "[o x l u v] &nbsp [o f l u v]<br>[8 0 7 0 6 0 14 0 4 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[f o e l u] &nbsp [e o f l u]<br>[z x w x d x r x h x s x e x f x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[v p u x q] &nbsp [v p n x q]<br>[k x t x j x o x l x v x h x q x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[u v p n] &nbsp [u v p n x]<br>[h x r x d x s x e x f x u x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[g z y w] &nbsp [g z y m]<br>[b x z x w x h x r x d x s x e x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[h i r k] &nbsp [h i r d]<br>[t t x x j j x x o o x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[o f l a v] &nbsp [o f l u v]<br>[n v u o j t k i m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[e o f l l u] &nbsp [e o f l u]<br>[3 3 4 4 5 5 6 6 6 6 7 7 8 8 9 9 10 10 11 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[w h m i r] &nbsp [w m h i r]<br>[21 23 25 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[o o f l u v] &nbsp [o f l u v]<br>[11 11 13 13 13 13 15 15 17 17 19 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split5": [
      {
        "prompt": "[h i r k] &nbsp [h i r d]<br>[u u f f e e s s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[o f l u] &nbsp [o f l v]<br>[24 22 20 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[p n x q q c] &nbsp [p n x q c]<br>[21 19 19 17 15 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[z m w y h] &nbsp [z y w m h]<br>[10 4 6 8 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[e o u l f] &nbsp [e o f l u]<br>[3 3 5 5 4 4 6 6 7 7 8 8 9 9 10 10 11 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[e j s o f] &nbsp [s j e o f]<br>[11 0 13 0 12 0 14 0 10 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[k d t s] &nbsp [k d t j]<br>[18 18 20 20 22 22 24 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[v p n x] &nbsp [v p n x q]<br>[v x u x l x f x o x e x j x s x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[o f l u] &nbsp [o f l u v]<br>[14 14 16 16 18 18 20 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[v p n x] &nbsp [u p n x]<br>[15 15 0 0 16 16 0 0 17 17 0 0 18 18 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split6": [
      {
        "prompt": "[j e o f] &nbsp [s e o f]<br>[k k t t j j o o l l v v n n q q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[d t s j] &nbsp [d t s e]<br>[9 8 7 6 5 4 3 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[e j s o f] &nbsp [s j e o f]<br>[a x k x y x m x i x g x t x j x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[l u v p j] &nbsp [l u v p n]<br>[13 13 15 15 7 7 19 19 21 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[h i r e d] &nbsp [h i r k d]<br>[13 0 15 0 7 0 19 0 21 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[m h i r] &nbsp [w h i r]<br>[12 0 13 0 14 0 15 0 16 0 17 0 18 0 19 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[d t s j j e] &nbsp [d t s j e]<br>[26 26 24 22 20 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[o o f l u v] &nbsp [o f l u v]<br>[22 22 0 0 22 22 0 0 23 23 0 0 24 24 0 0 25 25 0 0 26 26 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[o f l u] &nbsp [o f l u v]<br>[24 24 23 23 22 22 21 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[v p n x] &nbsp [v p n q]<br>[o o e e j j s s t t d d k k r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gensplit7": [
      {
        "prompt": "[s j e o] &nbsp [t j e o]<br>[23 23 0 0 24 24 0 0 25 25 0 0 26 26 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[z y w m] &nbsp [z y w m h]<br>[w w h h r r d d s s e e f f u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[m h i r] &nbsp [m h i r k]<br>[13 0 14 0 15 0 16 0 17 0 18 0 19 0 20 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[d t j s e] &nbsp [d t s j e]<br>[i x e x j x s x t x d x k x r x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[g z y w] &nbsp [g z y w m]<br>[c x x x p x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[b j z y w] &nbsp [b g z y w]<br>[7 7 6 6 5 5 18 18 3 3] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[v p n x] &nbsp [v p n x q]<br>[i x m x y x g x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[f l u v] &nbsp [f l u p]<br>[u x f x e x s x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[o f l k v] &nbsp [o f l u v]<br>[e e x x f f x x u u x x p p x x b b x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[v p n x] &nbsp [v p n x q]<br>[26 24 22 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ]
  },
  "alph_2": {
    "shuffled_letters": [
      "b",
      "n",
      "f",
      "m",
      "p",
      "g",
      "h",
      "w",
      "q",
      "e",
      "d",
      "l",
      "y",
      "s",
      "o",
      "c",
      "k",
      "a",
      "r",
      "j"
    ],
    "shuffled_alphabet": [
      "b",
      "n",
      "f",
      "m",
      "p",
      "g",
      "h",
      "w",
      "i",
      "q",
      "e",
      "d",
      "l",
      "y",
      "s",
      "o",
      "c",
      "k",
      "a",
      "t",
      "u",
      "v",
      "r",
      "x",
      "j",
      "z"
    ],
    "larger_int": [
      {
        "prompt": "[n f m p] &nbsp [n f m p g]<br>[q d y o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[l y c o s] &nbsp [l y s o c]<br>[e l a c s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[u v r x] &nbsp [t v r x]<br>[u r j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[i w q e d] &nbsp [w i q e d]<br>[h s e l i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[o o c k a t] &nbsp [o c k a t]<br>[f p h i e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[n f m p g g] &nbsp [n f m p g]<br>[d y o k t t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[t u v r x x] &nbsp [t u v r x]<br>[b f p h h i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[b n f m] &nbsp [b n f p]<br>[n m g w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[f m p g] &nbsp [f m p h]<br>[w q d y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[h w i q] &nbsp [h w i e]<br>[b f p h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "longer_targ": [
      {
        "prompt": "[l y s o] &nbsp [l y s c]<br>[c k a t u v r x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[o l y s d] &nbsp [d l y s o]<br>[m p i h w g q e d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[v r x j] &nbsp [v r x j z]<br>[c k a t u v r x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[a t u v] &nbsp [a t u r]<br>[m p g h w i q e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[l y j o c] &nbsp [l y s o c]<br>[b n f m p g a w i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[p g h w] &nbsp [p g h i]<br>[b n f m p g h w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[k a t u] &nbsp [k a t v]<br>[b n f m p g h w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[o c k a] &nbsp [o c k t]<br>[g h w i q e d l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[t u v j x] &nbsp [t u v r x]<br>[b n f m p g y w i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[i q e d] &nbsp [w q e d]<br>[l y s o c k a t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "group": [
      {
        "prompt": "[m p g h] &nbsp [m p g h w]<br>[t t u u v v r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[n f z p g] &nbsp [n f m p g]<br>[e e d d l l w w s s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[u v r x] &nbsp [u v r j]<br>[w w i i q q e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[e d l l y s] &nbsp [e d l y s]<br>[y y s s s s o o c c k k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[b n f m p p] &nbsp [b n f m p]<br>[u u v v r r x x x x j j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[y s o c] &nbsp [y s o k]<br>[c c k k a a t t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[w i q e] &nbsp [h i q e]<br>[g g h h w w i i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[k a t v u] &nbsp [k a t u v]<br>[f f g g p p m m h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[c k a t] &nbsp [c k a u]<br>[l l y y s s o o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[w a q e d] &nbsp [w i q e d]<br>[m m p p g g k k w w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "interleaved": [
      {
        "prompt": "[i q e h l] &nbsp [i q e d l]<br>[u x v x r x x x m x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[m p g h] &nbsp [f p g h]<br>[y x s x o x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[m p g h w w] &nbsp [m p g h w]<br>[n x f x m x p x g x g x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[o c k a] &nbsp [s c k a]<br>[t x u x v x r x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[m m p g h w] &nbsp [m p g h w]<br>[k x a x t x u x v x v x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[a t u v] &nbsp [a t u v r]<br>[g x h x w x i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[a k c t u] &nbsp [c k a t u]<br>[d x i x q x e x w x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[g h w i] &nbsp [p h w i]<br>[i x q x e x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[b n f m] &nbsp [b n f m p]<br>[k x a x t x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[b m f n p] &nbsp [b n f m p]<br>[g x h x w x q x i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "letter2num": [
      {
        "prompt": "[e d l y] &nbsp [e d l s]<br>[1 2 3 4] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[c k a t] &nbsp [c k a u]<br>[21 22 23 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[e q l y s] &nbsp [e d l y s]<br>[8 9 10 16 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[f f m p g h] &nbsp [f m p g h]<br>[9 10 11 12 13 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[t t u v r x] &nbsp [t u v r x]<br>[19 20 20 21 22 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[n f m p] &nbsp [b f m p]<br>[12 13 14 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[a v u t r] &nbsp [a t u v r]<br>[9 10 13 12 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[k c a t u] &nbsp [c k a t u]<br>[6 5 7 8 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[p g h h w i] &nbsp [p g h w i]<br>[4 5 5 6 7 8] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[d l y s] &nbsp [e l y s]<br>[14 15 16 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "reverse": [
      {
        "prompt": "[s o o c k a] &nbsp [s o c k a]<br>[i w h g p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[n f w p g] &nbsp [n f m p g]<br>[k v u t a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[t p g h w] &nbsp [m p g h w]<br>[h b p m f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[y s o c] &nbsp [y s o c k]<br>[r v u t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[c k a t] &nbsp [o k a t]<br>[u t a k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[q e d l] &nbsp [q e d l y]<br>[v u t a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[h w i q] &nbsp [h w i e]<br>[i w h g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[w i q e] &nbsp [w i q d]<br>[c o s y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[i q e d] &nbsp [w q e d]<br>[k c o s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[s o c k] &nbsp [s o c a]<br>[l d e q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split1": [
      {
        "prompt": "[g h w b q] &nbsp [g h w i q]<br>[i i e e l l o o c c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[s o c k] &nbsp [y o c k]<br>[13 0 14 0 15 0 16 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[f i p g h] &nbsp [f m p g h]<br>[15 17 19 21 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[m l y s o] &nbsp [d l y s o]<br>[q q d d y y o o z z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[l l y s o c] &nbsp [l y s o c]<br>[17 17 18 18 19 19 20 20 20 20 21 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[i q e d] &nbsp [i q e l]<br>[5 6 7 8 9 10 11 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[n f m p] &nbsp [n f m g]<br>[11 13 15 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[r x j z] &nbsp [v x j z]<br>[10 11 12 13 14 15 16 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[y s o c] &nbsp [y s o c k]<br>[9 11 13 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[n f m p] &nbsp [n f m g]<br>[a x k x c x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split2": [
      {
        "prompt": "[g n f m p] &nbsp [b n f m p]<br>[g g p p m m y y n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[d l y y s o] &nbsp [d l y s o]<br>[h x g x p x m x f x f x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[b n m f p] &nbsp [b n f m p]<br>[l a c s u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[q e d d l y] &nbsp [q e d l y]<br>[i i x x i i x x q q x x e e x x d d x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[k a t u] &nbsp [k a t u v]<br>[5 6 7 8 9 10 11 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[h w i q] &nbsp [h w i e]<br>[2 3 4 5 6 7 8 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[m p g h] &nbsp [m p g w]<br>[14 0 15 0 16 0 17 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[e w i q h] &nbsp [h w i q e]<br>[p p b b f f n n m m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[a t u v] &nbsp [a t u r]<br>[t t x x u u x x v v x x r r x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[b n f m p p] &nbsp [b n f m p]<br>[2 0 3 0 4 0 5 0 5 0 6 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split3": [
      {
        "prompt": "[b n f m] &nbsp [b n f p]<br>[d x l x y x s x o x c x k x a x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[u j r x v] &nbsp [u v r x j]<br>[15 0 16 0 17 0 19 0 18 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[w l y s o] &nbsp [d l y s o]<br>[i i w w h h g g n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[o c k a] &nbsp [o c k t]<br>[d d x x l l x x y y x x s s x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[k a t u v v] &nbsp [k a t u v]<br>[c c x x k k x x a a x x t t x x u u x x u u x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[p g s w i] &nbsp [p g h w i]<br>[10 11 12 13 14 22 16 17 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[n u v r x] &nbsp [t u v r x]<br>[16 3 5 7 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[g h w i] &nbsp [p h w i]<br>[o x s x y x l x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[p g h w] &nbsp [p g h i]<br>[7 7 8 8 9 9 10 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[n f m p] &nbsp [b f m p]<br>[o y d q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split4": [
      {
        "prompt": "[m p g g h w] &nbsp [m p g h w]<br>[s x o x c x k x a x t x u x v x r x r x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[k u t a v] &nbsp [k a t u v]<br>[d d l l a a s s o o c c k k y y t t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[h w i q] &nbsp [h w i e]<br>[15 16 17 18 19 20 21 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[m p g h] &nbsp [m p g w]<br>[u u x x v v x x r r x x x x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[w d q e i] &nbsp [w i q e d]<br>[6 5 4 7 8 9 10 11 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[b n f m] &nbsp [b n f m p]<br>[k k x x a a x x t t x x u u x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[l y s o p] &nbsp [l y s o c]<br>[9 8 20 6 5] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[i q e d] &nbsp [i q e l]<br>[17 19 21 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[y s b c k] &nbsp [y s o c k]<br>[c c a a u u r r e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[j f m p g] &nbsp [n f m p g]<br>[b b n n f f m m p p g g h h w w q q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split5": [
      {
        "prompt": "[y s o o c k] &nbsp [y s o c k]<br>[7 7 8 8 9 9 10 10 11 11 11 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[d l y s k] &nbsp [d l y s o]<br>[l x d x f x q x i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[e s l y d] &nbsp [e d l y s]<br>[19 19 16 16 17 17 18 18 15 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[p g h w] &nbsp [p g h w i]<br>[u x t x a x k x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[p g h w q] &nbsp [p g h w i]<br>[13 8 10 12 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[w d q e i] &nbsp [w i q e d]<br>[10 4 6 8 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[n f m p] &nbsp [n f m g]<br>[k x t x v x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[l y s o] &nbsp [d y s o]<br>[16 17 18 19 20 21 22 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[h w i q] &nbsp [h w i q e]<br>[k c o s y l d e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[i q e d m] &nbsp [i q e d l]<br>[e e b b l l y y s s o o c c k k a a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split6": [
      {
        "prompt": "[i q e d] &nbsp [w q e d]<br>[l s c a u r j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[s o a k c] &nbsp [s o c k a]<br>[w w g g m m q q d d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[i q e d] &nbsp [i q e d l]<br>[6 0 7 0 8 0 9 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[d l y s] &nbsp [e l y s]<br>[u u t t a a k k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[s o c k] &nbsp [s o c k a]<br>[m g w q d y o k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[e d l y] &nbsp [q d l y]<br>[s s x x o o x x c c x x k k x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[h w i q] &nbsp [h w i e]<br>[q x i x w x h x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[o t k a c] &nbsp [o c k a t]<br>[c u a r s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[n f m m p g] &nbsp [n f m p g]<br>[17 19 19 21 23 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[g i w h q] &nbsp [g h w i q]<br>[s s l l y y d d e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gensplit7": [
      {
        "prompt": "[n f g p m] &nbsp [n f m p g]<br>[o d y q k t v x z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[t u r v x] &nbsp [t u v r x]<br>[9 9 13 13 11 11 12 12 10 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[s a c k o] &nbsp [s o c k a]<br>[c x r x u x a x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[y s o o c k] &nbsp [y s o c k]<br>[7 9 9 11 13 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[h q i w e] &nbsp [h w i q e]<br>[9 7 8 6 5] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[y s o c c k] &nbsp [y s o c k]<br>[16 16 17 17 18 18 19 19 20 20 20 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[b n f m] &nbsp [b n f p]<br>[8 0 9 0 10 0 11 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[m p h g w] &nbsp [m p g h w]<br>[u a s c l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[q e d l] &nbsp [q e d l y]<br>[15 17 19 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[b n f m] &nbsp [b n f m p]<br>[8 8 9 9 10 10 11 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split1": [
      {
        "prompt": "[t u v r] &nbsp [t u v x]<br>[5 5 4 4 3 3 2 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[h w i q] &nbsp [h w i e]<br>[k x o x y x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[d l s y o] &nbsp [d l y s o]<br>[w w x x i i x x h h x x g g x x p p x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[p i h w g] &nbsp [p g h w i]<br>[n x y x g x w x q x d x m x o x k x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[b n f m] &nbsp [b n f p]<br>[f f x x p p x x h h x x i i x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[l y s o o c] &nbsp [l y s o c]<br>[t t a a k k c c o o s s s s y y l l d d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[v r x j] &nbsp [v r x z]<br>[12 12 11 11 10 10 9 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[b n f m] &nbsp [b n f p]<br>[7 9 11 13 15 17 19 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[f m p g] &nbsp [n m p g]<br>[19 0 18 0 17 0 16 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[l y o s c] &nbsp [l y s o c]<br>[23 0 21 0 22 0 20 0 19 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split2": [
      {
        "prompt": "[h w i q] &nbsp [h w i q e]<br>[c x s x l x e x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[e d d l y s] &nbsp [e d l y s]<br>[f f x x m m x x p p x x g g x x h h x x w w x x i i x x q q x x q q x x e e x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[y o s c k] &nbsp [y s o c k]<br>[12 16 14 18 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[i i q e d l] &nbsp [i q e d l]<br>[9 8 7 7 6 5 4 3 2 1] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[y s s o c k] &nbsp [y s o c k]<br>[l x e x e x i x h x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[x n f m p] &nbsp [b n f m p]<br>[7 0 6 0 5 0 19 0 3 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[e d l y] &nbsp [e d l s]<br>[v v u u t t a a k k c c o o s s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[q e d l] &nbsp [i e d l]<br>[e e x x d d x x l l x x y y x x s s x x o o x x c c x x k k x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[b n f m] &nbsp [b n f m p]<br>[7 9 11 13 15 17 19 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[c k a u t] &nbsp [c k a t u]<br>[11 11 13 13 19 19 17 17 15 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split3": [
      {
        "prompt": "[c k u t a] &nbsp [c k a t u]<br>[10 0 12 0 16 0 14 0 18 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[v r x j] &nbsp [u r x j]<br>[19 19 20 20 21 21 22 22 23 23 24 24 25 25 26 26] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[w i q e d d] &nbsp [w i q e d]<br>[p p x x h h x x h h x x i i x x e e x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[f m p g] &nbsp [n m p g]<br>[o o x x c c x x k k x x a a x x t t x x u u x x v v x x r r x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[q e e d l y] &nbsp [q e d l y]<br>[16 16 0 0 17 17 0 0 17 17 0 0 18 18 0 0 19 19 0 0 20 20 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[t u v r] &nbsp [t u v r x]<br>[2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[c k a t] &nbsp [o k a t]<br>[a c s l e i h p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[d l y s] &nbsp [e l y s]<br>[u a c s l e i h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[v r x j] &nbsp [v r x j z]<br>[i i e e l l s s c c a a u u r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[s o c k] &nbsp [s o c k a]<br>[17 16 15 14 13 12 11 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split4": [
      {
        "prompt": "[g h w i] &nbsp [p h w i]<br>[11 11 0 0 12 12 0 0 13 13 0 0 14 14 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[q e d l] &nbsp [q e d y]<br>[2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[w i d e q] &nbsp [w i q e d]<br>[15 21 19 17 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[d l y s] &nbsp [d l y s o]<br>[l l x x d d x x e e x x q q x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[l y s o] &nbsp [l y s o c]<br>[a c s l e i h p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[n f m p] &nbsp [n f m p g]<br>[h h x x w w x x i i x x q q x x e e x x d d x x l l x x y y x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[r x j z] &nbsp [v x j z]<br>[n n x x f f x x m m x x p p x x g g x x h h x x w w x x i i x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[p g w h i] &nbsp [p g h w i]<br>[17 17 0 0 18 18 0 0 19 19 0 0 21 21 0 0 20 20 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[q e d o y] &nbsp [q e d l y]<br>[f x p x h x i x e x t x s x c x a x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[s o c k a a] &nbsp [s o c k a]<br>[2 2 2 2 4 4 6 6 8 8 10 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split5": [
      {
        "prompt": "[r x j z] &nbsp [v x j z]<br>[d d q q w w g g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[f m p g b] &nbsp [f m p g h]<br>[18 17 4 15 14 13 12 11 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[i q e d] &nbsp [w q e d]<br>[d x y x o x k x t x v x x x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[y s o c] &nbsp [y s o c k]<br>[o o x x s s x x y y x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[u k a t c] &nbsp [c k a t u]<br>[e x i x h x f x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[i q e d] &nbsp [w q e d]<br>[8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[d s y l o] &nbsp [d l y s o]<br>[z x k t v o y d q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[b n f m] &nbsp [b n f p]<br>[7 7 0 0 8 8 0 0 9 9 0 0 10 10 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[g h w i] &nbsp [g h w q]<br>[21 19 17 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[l y s o] &nbsp [l y s o c]<br>[17 15 13 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split6": [
      {
        "prompt": "[a t g v r] &nbsp [a t u v r]<br>[f f x x p p x x r r x x i i x x e e x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[m p g h] &nbsp [m p g w]<br>[x x r r v v u u t t a a k k c c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[b b n f m p] &nbsp [b n f m p]<br>[2 2 2 2 4 4 6 6 8 8 10 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[p g h w] &nbsp [m g h w]<br>[s s x x y y x x l l x x d d x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[b n n f m p] &nbsp [b n f m p]<br>[4 4 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[v r x j] &nbsp [v r x j z]<br>[o o x x s s x x y y x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[f m p g] &nbsp [n m p g]<br>[j x x x r x v x u x t x a x k x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[b n f m p p] &nbsp [b n f m p]<br>[16 15 14 13 12 11 11 10 9 8] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[r x j z] &nbsp [v x j z]<br>[5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[w p g h m] &nbsp [m p g h w]<br>[1 11 5 7 9 3 13 15 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gensplit7": [
      {
        "prompt": "[r x j z] &nbsp [v x j z]<br>[h h x x i i x x e e x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[f m p h g] &nbsp [f m p g h]<br>[c c k k a a o o s s y y l l d d e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[y s o c] &nbsp [y s o c k]<br>[9 0 11 0 13 0 15 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[o c k v t] &nbsp [o c k a t]<br>[p p h h i i e e y y s s c c a a u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[f m p g] &nbsp [f m p h]<br>[d d e e q q i i w w h h g g p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[g h w i] &nbsp [p h w i]<br>[g g x x w w x x q q x x d d x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[o c k k a t] &nbsp [o c k a t]<br>[4 4 6 6 6 6 8 8 10 10 12 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[k a t t u v] &nbsp [k a t u v]<br>[1 1 2 2 3 3 4 4 4 4 5 5 6 6 7 7 8 8 9 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[q e d l] &nbsp [q e d y]<br>[f x p x h x i x e x l x s x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[n f m p] &nbsp [n f m g]<br>[a x k x c x o x s x y x l x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ]
  },
  "alph_3": {
    "shuffled_letters": [
      "i",
      "s",
      "f",
      "z",
      "v",
      "t",
      "c",
      "r",
      "o",
      "g",
      "e",
      "w",
      "j",
      "m",
      "x",
      "q",
      "n",
      "y",
      "a",
      "k"
    ],
    "shuffled_alphabet": [
      "i",
      "b",
      "s",
      "d",
      "f",
      "z",
      "v",
      "h",
      "t",
      "c",
      "r",
      "l",
      "o",
      "g",
      "e",
      "p",
      "w",
      "j",
      "m",
      "x",
      "u",
      "q",
      "n",
      "y",
      "a",
      "k"
    ],
    "larger_int": [
      {
        "prompt": "[j m x u] &nbsp [j m x q]<br>[v t r o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[w j m x] &nbsp [w j m x u]<br>[f v t r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[d f z v] &nbsp [d f z v h]<br>[w m u n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[o g d p w] &nbsp [o g e p w]<br>[e w s u n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[z v h t] &nbsp [z v h c]<br>[i s f v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[w j m x] &nbsp [w j m u]<br>[b d z h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[l o g e] &nbsp [l o g e p]<br>[f v t r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[i b b s d f] &nbsp [i b s d f]<br>[o e w m m u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[h t c r] &nbsp [v t c r]<br>[m u n a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[r l o g] &nbsp [c l o g]<br>[j x q y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "longer_targ": [
      {
        "prompt": "[p e g w j] &nbsp [g e p w j]<br>[h b s d f z v i t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[d f z v] &nbsp [d f z v h]<br>[r l o g e p w j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[u q n y] &nbsp [u q n a]<br>[i b s d f z v h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[m x u q] &nbsp [m x u n]<br>[i b s d f z v h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[s d f z] &nbsp [b d f z]<br>[b s d f z v h t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[l o g e] &nbsp [l o g p]<br>[c r l o g e p w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[q n y a k k] &nbsp [q n y a k]<br>[b s d f z z v h t c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[p p w j m x] &nbsp [p w j m x]<br>[s d f z v h t c c r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[u x q n y] &nbsp [x u q n y]<br>[r l o g e p w m j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[m x u q] &nbsp [m x u n]<br>[f z v h t c r l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "group": [
      {
        "prompt": "[q n y a] &nbsp [q n y a k]<br>[o o g g e e p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[v h t c c r] &nbsp [v h t c r]<br>[u u q q n n y y a a a a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[h c t r l] &nbsp [h t c r l]<br>[c c g g l l o o r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[w e p g j] &nbsp [g e p w j]<br>[f f d d z z v v h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[l o g e] &nbsp [r o g e]<br>[q q n n y y a a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[o p e g w] &nbsp [o g e p w]<br>[q q a a y y n n k k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[c r l o] &nbsp [c r l o g]<br>[o o g g e e p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[x u q n] &nbsp [x u q y]<br>[o o g g e e p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[p g e o w] &nbsp [o g e p w]<br>[q q n n y y k k a a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[m e u q n] &nbsp [m x u q n]<br>[u u q q v v y y a a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "interleaved": [
      {
        "prompt": "[q n y a] &nbsp [q n y k]<br>[i x b x s x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[p x j m w] &nbsp [p w j m x]<br>[c x g x l x o x r x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[z v h t] &nbsp [z v h c]<br>[m x x x u x q x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[r l o g] &nbsp [r l o g e]<br>[c x r x l x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[o g e p] &nbsp [o g e p w]<br>[q x n x y x a x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[v e p w j] &nbsp [g e p w j]<br>[r x j x o x g x e x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[x u q n] &nbsp [m u q n]<br>[o x g x e x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[p w x m j] &nbsp [p w j m x]<br>[v x z x f x h x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[v f z d h] &nbsp [d f z v h]<br>[i x s x b x d x f x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[c r l o] &nbsp [c r l o g]<br>[v x h x t x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "letter2num": [
      {
        "prompt": "[u q n y] &nbsp [u q n y a]<br>[1 2 3 4] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[g e p w] &nbsp [g e p j]<br>[4 5 6 7] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[z h v t c] &nbsp [z v h t c]<br>[21 18 19 20 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[e p w j f] &nbsp [e p w j m]<br>[20 21 22 23 3] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[x u q l y] &nbsp [x u q n y]<br>[14 15 7 17 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[c r l o] &nbsp [c r l g]<br>[1 2 3 4] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[u q n s a] &nbsp [u q n y a]<br>[15 16 23 18 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[q n y a] &nbsp [q n y k]<br>[19 20 21 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[h t c r] &nbsp [v t c r]<br>[20 21 22 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[l o g e] &nbsp [l o g e p]<br>[1 2 3 4] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "reverse": [
      {
        "prompt": "[i b s c f] &nbsp [i b s d f]<br>[t h v z j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[p w j m] &nbsp [e w j m]<br>[p e g o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[c r l o] &nbsp [c r l g]<br>[n q u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[n y a k] &nbsp [q y a k]<br>[h v z f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[q n y a] &nbsp [q n y k]<br>[o l r c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[h f z v d] &nbsp [d f z v h]<br>[v z s d f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[h t c r] &nbsp [v t c r]<br>[l r c t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[e p w j] &nbsp [g p w j]<br>[n q u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[d f z z v h] &nbsp [d f z v h]<br>[u u x m j w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[q n y a] &nbsp [q n y k]<br>[g o l r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split1": [
      {
        "prompt": "[i b s d v] &nbsp [i b s d f]<br>[v x t x d x o x e x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[i b s d] &nbsp [i b s d f]<br>[j x m x x x u x q x n x y x a x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[u x n y a] &nbsp [u q n y a]<br>[j x x x q x y x w x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[e g p w j] &nbsp [g e p w j]<br>[9 10 17 12 13 14 15 16 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[m x u q] &nbsp [j x u q]<br>[17 16 15 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[r l o g] &nbsp [r l o g e]<br>[k a y n q u x m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[m x u q] &nbsp [m x u n]<br>[v x h x t x c x r x l x o x g x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[h d c r l] &nbsp [h t c r l]<br>[v v x x x x x x u u x x q q x x n n x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[h t c r] &nbsp [h t c l]<br>[15 0 16 0 17 0 18 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[k n y a q] &nbsp [q n y a k]<br>[15 13 11 17 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split2": [
      {
        "prompt": "[h h t c r l] &nbsp [h t c r l]<br>[e x p x w x j x j x m x x x u x q x n x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[e p w j q] &nbsp [e p w j m]<br>[a n b m w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[s d f f z v] &nbsp [s d f z v]<br>[17 16 16 15 14 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[l l o g e p] &nbsp [l o g e p]<br>[7 8 9 10 11 12 12 13 14 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[i b s d v] &nbsp [i b s d f]<br>[v x t x r x k x e x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[i b s d d f] &nbsp [i b s d f]<br>[b b s s d d f f f f z z v v h h t t c c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[u q n y y a] &nbsp [u q n y a]<br>[23 22 21 20 19 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[i b s d] &nbsp [i b s f]<br>[z x v x h x t x c x r x l x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[i b s d f f] &nbsp [i b s d f]<br>[12 14 16 18 18 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[m x u q] &nbsp [j x u q]<br>[w w p p e e g g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split3": [
      {
        "prompt": "[b s d f z z] &nbsp [b s d f z]<br>[i x i x b x s x d x f x z x v x h x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[l o g e e p] &nbsp [l o g e p]<br>[18 17 16 16 15 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[t c r l o o] &nbsp [t c r l o]<br>[m x m x u x n x a x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[f z v h t t] &nbsp [f z v h t]<br>[1 2 2 3 4 5 6 7 8 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[t c r l] &nbsp [t c r o]<br>[17 16 15 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[h t c r r l] &nbsp [h t c r l]<br>[o x o x l x r x c x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[z v t h c] &nbsp [z v h t c]<br>[17 19 23 21 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[q n y k a] &nbsp [q n y a k]<br>[5 4 3 1 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[c r r l o g] &nbsp [c r l o g]<br>[o x e x w x m x m x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[z v h t] &nbsp [z v h c]<br>[n n q q u u x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split4": [
      {
        "prompt": "[p w j m] &nbsp [p w j x]<br>[e g o l r c t h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[q n n y a k] &nbsp [q n y a k]<br>[14 15 16 17 17 18 19 20 21 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[f z w h t] &nbsp [f z v h t]<br>[c l g p j x q s k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[l o g b p] &nbsp [l o g e p]<br>[22 0 23 0 24 0 25 0 5 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[u q n y] &nbsp [u q n a]<br>[c c t t h h v v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[t c r l] &nbsp [t c r o]<br>[d x z x h x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[b s d f] &nbsp [b s d z]<br>[o o x x g g x x e e x x p p x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[i b s d w] &nbsp [i b s d f]<br>[o o c c l l g g p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[b s d f] &nbsp [b s d z]<br>[18 18 19 19 20 20 21 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[q n y a k k] &nbsp [q n y a k]<br>[q x j p p g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split5": [
      {
        "prompt": "[z v h t c c] &nbsp [z v h t c]<br>[w w j j j j m m x x u u q q n n y y a a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[m j w x u] &nbsp [w j m x u]<br>[10 16 14 12 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[z v h t] &nbsp [z v h c]<br>[y x n x q x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[v h t c] &nbsp [v h t c r]<br>[17 19 21 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[b s j f z] &nbsp [b s d f z]<br>[6 0 7 0 8 0 9 0 5 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[g e p w] &nbsp [g e p j]<br>[16 15 14 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[p w j m] &nbsp [p w j x]<br>[r r x x l l x x o o x x g g x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[x q u n y] &nbsp [x u q n y]<br>[o o l l g g e e p p w w j j m m x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[f z v h] &nbsp [f z v t]<br>[e e p p w w j j m m x x u u q q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[m x u q n n] &nbsp [m x u q n]<br>[h x h x c x l x g x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split6": [
      {
        "prompt": "[e p w j] &nbsp [g p w j]<br>[x x q x y x k x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[k f z v h] &nbsp [d f z v h]<br>[w w m m e e g g o o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[b s d d f z] &nbsp [b s d f z]<br>[10 0 11 0 12 0 13 0 14 0 14 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[q n y a] &nbsp [q n y k]<br>[d x z x h x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[g e p w] &nbsp [o e p w]<br>[j x m x x x u x q x n x y x a x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[f z v h] &nbsp [f z v t]<br>[20 20 21 21 22 22 23 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[v h t c] &nbsp [z h t c]<br>[22 22 23 23 24 24 25 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[m x u q] &nbsp [j x u q]<br>[r r c c t t h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[u q n a y] &nbsp [u q n y a]<br>[10 3 4 5 6 7 8 9 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[f z v h] &nbsp [d z v h]<br>[x x x x u u x x q q x x n n x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gensplit7": [
      {
        "prompt": "[r l o o g e] &nbsp [r l o g e]<br>[18 18 19 19 19 19 20 20 21 21 22 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[j m x f q] &nbsp [j m x u q]<br>[o x a x w x m x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[d f z v] &nbsp [s f z v]<br>[r r o o e e w w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[w j m x] &nbsp [w j m u]<br>[v t r o e w m u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[z v h t] &nbsp [z v h c]<br>[k a y n q u x m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[k n y a q] &nbsp [q n y a k]<br>[o x x x e x p x w x j x m x g x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[r l o g] &nbsp [c l o g]<br>[u x x x m x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[m x u q] &nbsp [m x u n]<br>[18 0 19 0 20 0 21 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[m x u q] &nbsp [m x u n]<br>[z h c l g p j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[z v h t] &nbsp [z v h c]<br>[10 12 14 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split1": [
      {
        "prompt": "[w j m x] &nbsp [w j m u]<br>[12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[n y a k] &nbsp [q y a k]<br>[18 18 0 0 19 19 0 0 20 20 0 0 21 21 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[o g e p] &nbsp [l g e p]<br>[t x v x f x s x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[j m x u] &nbsp [j m x q]<br>[23 22 21 20 19 18 17 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[x u q n f] &nbsp [x u q n y]<br>[a x s x u x m x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[q n y a] &nbsp [u n y a]<br>[h h v v z z f f d d s s b b i i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[i b s d] &nbsp [i b s f]<br>[20 0 19 0 18 0 17 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[n u q x y] &nbsp [x u q n y]<br>[22 21 20 14 18 17 16 15 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[z v h t] &nbsp [z v h t c]<br>[o x l x r x c x t x h x v x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[j m x u] &nbsp [j m x q]<br>[26 0 25 0 24 0 23 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split2": [
      {
        "prompt": "[c r l o] &nbsp [c r l g]<br>[w e o r t v f s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[q n y a] &nbsp [q n y k]<br>[z x h x c x l x g x p x j x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[m x x u q n] &nbsp [m x u q n]<br>[f f v v t t r r o o e e w w w w m m u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[i b s d] &nbsp [i b s f]<br>[18 18 20 20 22 22 24 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[w j m x] &nbsp [w j m u]<br>[11 10 9 8 7 6 5 4] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[q n y a] &nbsp [q n y k]<br>[24 0 23 0 22 0 21 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[q n l a k] &nbsp [q n y a k]<br>[25 23 21 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[z v h t] &nbsp [z v h t c]<br>[c c l l g g p p j j x x q q y y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[g e p p w j] &nbsp [g e p w j]<br>[j j x x m m x x x x x x u u x x q q x x n n x x y y x x a a x x a a x x k k x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[t h v c r] &nbsp [v h t c r]<br>[l l w w p p e e g g o o j j r r c c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split3": [
      {
        "prompt": "[r l o g] &nbsp [r l o e]<br>[8 0 9 0 10 0 11 0 12 0 13 0 14 0 15 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[x u q n] &nbsp [x u q y]<br>[b b d d z z h h c c l l g g p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[t c r l] &nbsp [t c r o]<br>[18 16 14 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[u q n y] &nbsp [u q n y a]<br>[o o x x e e x x w w x x m m x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[d z f v h] &nbsp [d f z v h]<br>[k k x x x x x x q q x x y y x x j j x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[s d f z] &nbsp [b d f z]<br>[h x v x z x f x d x s x b x i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[c r l o] &nbsp [c r l o g]<br>[z z h h c c l l g g p p j j x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[e p w j] &nbsp [g p w j]<br>[f f x x v v x x t t x x r r x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[z s d f b] &nbsp [b s d f z]<br>[v v x x e e x x t t x x c c x x r r x x l l x x o o x x g g x x h h x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[q n y a] &nbsp [q n y k]<br>[1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split4": [
      {
        "prompt": "[e p m j w] &nbsp [e p w j m]<br>[u m e w o r t v f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[d o g e p] &nbsp [l o g e p]<br>[a x w x u x m x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[a n y q k] &nbsp [q n y a k]<br>[6 0 8 0 7 0 9 0 10 0 11 0 12 0 13 0 14 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[t c r l] &nbsp [t c r o]<br>[5 0 4 0 3 0 2 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[p w j m] &nbsp [e w j m]<br>[19 0 20 0 21 0 22 0 23 0 24 0 25 0 26 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[i b s m f] &nbsp [i b s d f]<br>[15 15 14 14 13 13 12 12 26 26] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[m x u q] &nbsp [m x u q n]<br>[9 0 11 0 13 0 15 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[z v h h t c] &nbsp [z v h t c]<br>[11 11 0 0 12 12 0 0 12 12 0 0 13 13 0 0 14 14 0 0 15 15 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[z z v h t c] &nbsp [z v h t c]<br>[r x r x t x v x f x s x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[u q n y i] &nbsp [u q n y a]<br>[7 9 11 22 15 17 19 21 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split5": [
      {
        "prompt": "[o g e p] &nbsp [l g e p]<br>[a a x x y y x x n n x x q q x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[e p w j j m] &nbsp [e p w j m]<br>[y x n x q x q x u x x x m x j x w x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[r l d g e] &nbsp [r l o g e]<br>[t x y x o x e x w x m x u x n x a x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[p w j m] &nbsp [p w j x]<br>[x x j j p p g g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[m p w j e] &nbsp [e p w j m]<br>[16 16 18 18 20 20 24 24 22 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[i b s d] &nbsp [i b s f]<br>[18 18 19 19 20 20 21 21 22 22 23 23 24 24 25 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[c r l o] &nbsp [c r l g]<br>[e e x x p p x x w w x x j j x x m m x x x x x x u u x x q q x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[u q n y] &nbsp [u q n a]<br>[f f x x v v x x t t x x r r x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[x u q n] &nbsp [x u q y]<br>[16 14 12 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[x u q n] &nbsp [m u q n]<br>[y q x j p g l c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split6": [
      {
        "prompt": "[f z v h] &nbsp [d z v h]<br>[g g o o l l r r c c t t h h v v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[c r l o] &nbsp [t r l o]<br>[12 0 14 0 16 0 18 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[o g g e p w] &nbsp [o g e p w]<br>[e x o x o x r x t x v x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[a n y q k] &nbsp [q n y a k]<br>[8 10 12 6 4] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[z v h t] &nbsp [z v h c]<br>[u m w e o r t v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[s d f f z v] &nbsp [s d f z v]<br>[w w x x j j x x m m x x x x x x x x x x u u x x q q x x n n x x y y x x a a x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[b s d f] &nbsp [i s d f]<br>[9 7 5 3] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[s d f z] &nbsp [b d f z]<br>[c x l x g x p x j x x x q x y x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[u q n y] &nbsp [u q n y a]<br>[17 0 19 0 21 0 23 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[d f z v] &nbsp [d f z h]<br>[9 11 13 15 17 19 21 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gensplit7": [
      {
        "prompt": "[b s d f] &nbsp [i s d f]<br>[v v x x h h x x t t x x c c x x r r x x l l x x o o x x g g x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[z v h c t] &nbsp [z v h t c]<br>[x x g x p x j x l x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[l o g w p] &nbsp [l o g e p]<br>[5 18 9 11 13 15 17 19 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[g g e p w j] &nbsp [g e p w j]<br>[24 23 22 21 20 19 19 18 17 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[z v h g c] &nbsp [z v h t c]<br>[c c x x e e x x g g x x p p x x j j x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[v h t c] &nbsp [v h t r]<br>[9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[g e p j w] &nbsp [g e p w j]<br>[o x e x w x r x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[t c r l] &nbsp [t c r o]<br>[n n q q u u x x m m j j w w p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[h w j m x] &nbsp [p w j m x]<br>[l l a a c c t t h h v v z z f f d d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[c r l o] &nbsp [t r l o]<br>[h h x x c c x x l l x x g g x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ]
  },
  "alph_4": {
    "shuffled_letters": [
      "w",
      "p",
      "z",
      "b",
      "a",
      "s",
      "j",
      "g",
      "x",
      "o",
      "m",
      "n",
      "t",
      "k",
      "f",
      "i",
      "q",
      "y",
      "r",
      "l"
    ],
    "shuffled_alphabet": [
      "w",
      "p",
      "c",
      "d",
      "e",
      "z",
      "b",
      "h",
      "a",
      "s",
      "j",
      "g",
      "x",
      "o",
      "m",
      "n",
      "t",
      "k",
      "f",
      "i",
      "u",
      "v",
      "q",
      "y",
      "r",
      "l"
    ],
    "larger_int": [
      {
        "prompt": "[a s j g x x] &nbsp [a s j g x]<br>[w c c e b a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[b h a s] &nbsp [z h a s]<br>[b a j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[s j g x] &nbsp [a j g x]<br>[t f u q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[e z b h h a] &nbsp [e z b h a]<br>[n k i i v y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[b h a s] &nbsp [z h a s]<br>[c e b a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[o m n t] &nbsp [o m n t k]<br>[p d z h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[t k f i j] &nbsp [t k f i u]<br>[k i v x l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[i u v q] &nbsp [i u v y]<br>[s g o n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[t k k f i u] &nbsp [t k f i u]<br>[j x m t f f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[v q y r] &nbsp [v q y r l]<br>[k i v y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "longer_targ": [
      {
        "prompt": "[e z b u a] &nbsp [e z b h a]<br>[z b h a s j g x w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[s j g x] &nbsp [s j g o]<br>[k f i u v q y r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[v q y r] &nbsp [v q y l]<br>[o m n t k f i u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[f i u v] &nbsp [f i u q]<br>[s j g x o m n t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[w p c d] &nbsp [w p c d e]<br>[m n t k f i u v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[d e z b] &nbsp [d e z b h]<br>[n t k f i u v q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[b h a s] &nbsp [b h a j]<br>[n t k f i u v q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[s u v q y] &nbsp [i u v q y]<br>[g x o e n t k f i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[h a s j] &nbsp [h a s g]<br>[t k f i u v q y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[v q y r] &nbsp [v q y r l]<br>[b h a s j g x o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "group": [
      {
        "prompt": "[k f i u] &nbsp [k f i v]<br>[c c d d e e z z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[d e z b] &nbsp [d e z h]<br>[e e z z b b h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[v q y r] &nbsp [v q y l]<br>[e e z z b b h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[w p c d] &nbsp [w p c d e]<br>[s s j j g g x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[o m n t] &nbsp [o m n k]<br>[t t k k f f i i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[a s j g] &nbsp [a s j x]<br>[j j g g x x o o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[s j g x] &nbsp [a j g x]<br>[u u v v q q y y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[v q y r] &nbsp [v q y l]<br>[s s j j g g x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[b h a s] &nbsp [b h a j]<br>[t t k k f f i i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[h a s j] &nbsp [h a s g]<br>[p p c c d d e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "interleaved": [
      {
        "prompt": "[f i u v] &nbsp [f i u q]<br>[u x v x q x y x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[a s j g] &nbsp [a s j x]<br>[x x o x m x n x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[k f i u v v] &nbsp [k f i u v]<br>[s x j x j x g x x x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[o m n t] &nbsp [o m n t k]<br>[t x k x f x i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[e z b a h] &nbsp [e z b h a]<br>[d x z x e x b x h x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[d e z w h] &nbsp [d e z b h]<br>[a x s x j x z x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[j g x o m m] &nbsp [j g x o m]<br>[b x b x h x a x s x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[p c d e] &nbsp [p c d e z]<br>[w x p x c x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[c d e z] &nbsp [c d e z b]<br>[w x p x c x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[g x o m n n] &nbsp [g x o m n]<br>[m x n x t x k x k x f x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "letter2num": [
      {
        "prompt": "[p c z e d] &nbsp [p c d e z]<br>[13 14 15 17 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[g x o m] &nbsp [g x o n]<br>[4 5 6 7] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[q y r l] &nbsp [v y r l]<br>[20 21 22 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[g x o m] &nbsp [g x o n]<br>[22 23 24 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[g x o m] &nbsp [g x o n]<br>[21 22 23 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[x x o m n t] &nbsp [x o m n t]<br>[1 2 2 3 4 5] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[i u v q] &nbsp [i u v q y]<br>[7 8 9 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[k f i u] &nbsp [t f i u]<br>[15 16 17 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[g x o m] &nbsp [g x o n]<br>[11 12 13 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[q y r l] &nbsp [v y r l]<br>[2 3 4 5] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "reverse": [
      {
        "prompt": "[u v q y] &nbsp [u v q r]<br>[k t n m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[j g x o k] &nbsp [j g x o m]<br>[g j f a h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[d e z b b h] &nbsp [d e z b h]<br>[f k t n n m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[c d e z] &nbsp [c d e z b]<br>[m o x g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[j g x o] &nbsp [j g x m]<br>[y q v u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[z b h a] &nbsp [e b h a]<br>[e d c p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[g x o m] &nbsp [g x o n]<br>[l r y q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[o m n t] &nbsp [x m n t]<br>[z e d c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[e z b h] &nbsp [e z b a]<br>[l r y q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[m n f k t] &nbsp [m n t k f]<br>[r l y q v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split1": [
      {
        "prompt": "[e z b h] &nbsp [e z b h a]<br>[o o x x g g j j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[t w f i u] &nbsp [t k f i u]<br>[i k n w g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[e z b h] &nbsp [e z b a]<br>[d d z z h h s s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[v q r y l] &nbsp [v q y r l]<br>[7 5 3 9 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[c d e z z b] &nbsp [c d e z b]<br>[z z x x b b x x b b x x h h x x a a x x s s x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[n t k f] &nbsp [n t k f i]<br>[1 0 2 0 3 0 4 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[o m n t] &nbsp [o m n k]<br>[f x k x t x n x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[j a s h g] &nbsp [h a s j g]<br>[19 15 17 13 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[g x o m] &nbsp [j x o m]<br>[j a b e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[x o e n t] &nbsp [x o m n t]<br>[c c x x d d x x r r x x z z x x b b x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split2": [
      {
        "prompt": "[t k f i] &nbsp [t k f i u]<br>[14 14 15 15 16 16 17 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[k f i u] &nbsp [k f i u v]<br>[j a b e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[j g x o] &nbsp [s g x o]<br>[12 12 13 13 14 14 15 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[p c d e e z] &nbsp [p c d e z]<br>[g s h z d d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[o m n t] &nbsp [o m n k]<br>[s s g g o o n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[q y r l] &nbsp [v y r l]<br>[b b x x h h x x a a x x s s x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[v q y r] &nbsp [u q y r]<br>[o n k i v y l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[m n t k] &nbsp [m n t f]<br>[w x p x c x d x e x z x b x h x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[n t k f] &nbsp [n t k i]<br>[w w p p c c d d e e z z b b h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[c h e z b] &nbsp [c d e z b]<br>[8 8 9 9 10 10 18 18 12 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split3": [
      {
        "prompt": "[x o m n] &nbsp [x o m t]<br>[p x c x d x e x z x b x h x a x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[a s j g] &nbsp [a s j x]<br>[t x f x u x q x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[b h a s] &nbsp [b h a s j]<br>[j a b e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[v q y r n] &nbsp [v q y r l]<br>[5 5 11 11 7 7 8 8 9 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[c d d e z b] &nbsp [c d e z b]<br>[l x l x r x y x q x v x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[k f i u g] &nbsp [k f i u v]<br>[a b e d w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[k f i u] &nbsp [t f i u]<br>[k t n m o x g j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[f i u v] &nbsp [k i u v]<br>[7 6 5 4] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[e o b h a] &nbsp [e z b h a]<br>[12 14 16 6 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[c d e z] &nbsp [p d e z]<br>[q q x x y y x x r r x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split4": [
      {
        "prompt": "[v q y r] &nbsp [u q y r]<br>[k k f f i i u u v v q q y y r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[n n t k f i] &nbsp [n t k f i]<br>[n n n n m m o o x x g g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[v q y r] &nbsp [v q y r l]<br>[s g o n k i v y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[m n t k] &nbsp [m n t f]<br>[26 25 24 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[q y r l] &nbsp [v y r l]<br>[e e d d c c p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[f f i u v q] &nbsp [f i u v q]<br>[22 22 22 22 23 23 24 24 25 25 26 26] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[c d e z] &nbsp [c d e z b]<br>[f f k k t t n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[g x o m] &nbsp [g x o n]<br>[18 19 20 21 22 23 24 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[s j g x] &nbsp [s j g o]<br>[h h x x a a x x s s x x j j x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[p c d e] &nbsp [w c d e]<br>[g o n k i v y l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split5": [
      {
        "prompt": "[u k f i t] &nbsp [t k f i u]<br>[m m t t f f q q u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[p c d e] &nbsp [p c d e z]<br>[6 8 10 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[g x o m] &nbsp [g x o m n]<br>[e x b x a x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[w p c d] &nbsp [w p c d e]<br>[3 4 5 6 7 8 9 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[y v q u r] &nbsp [u v q y r]<br>[13 13 14 14 16 16 15 15 17 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[g x o m] &nbsp [j x o m]<br>[i x v x y x l x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[j g x o] &nbsp [s g x o]<br>[6 7 8 9 10 11 12 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[c d e y b] &nbsp [c d e z b]<br>[c c x x q q x x y y x x r r x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[c d e z] &nbsp [c d e b]<br>[16 0 17 0 18 0 19 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[i u v q] &nbsp [i u v y]<br>[16 0 17 0 18 0 19 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split6": [
      {
        "prompt": "[s j g x] &nbsp [s j g x o]<br>[j x m t f u q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[f t k n i] &nbsp [n t k f i]<br>[g x x x i x m x n x t x k x f x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[i u v q] &nbsp [f u v q]<br>[8 7 6 5] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[q y r l] &nbsp [v y r l]<br>[j j x x m m t t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[s j g x] &nbsp [s j g o]<br>[n m o x g j s a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[e z b h] &nbsp [e z b h a]<br>[9 0 10 0 11 0 12 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[i u x q y] &nbsp [i u v q y]<br>[t x n x d x o x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[i u v q] &nbsp [f u v q]<br>[9 8 7 6] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[i u b q y] &nbsp [i u v q y]<br>[w w x x s s x x c c x x d d x x e e x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[e q y r l] &nbsp [v q y r l]<br>[x x o o t t f f u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gensplit7": [
      {
        "prompt": "[m n t k] &nbsp [o n t k]<br>[v v u u i i f f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[e p c d w] &nbsp [w p c d e]<br>[25 0 23 0 24 0 22 0 26 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[n t k f] &nbsp [n t k f i]<br>[k n o g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[i e v q y] &nbsp [i u v q y]<br>[n m o x g j s a d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[u v q y] &nbsp [i v q y]<br>[f x i x u x v x q x y x r x l x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[h j s a g] &nbsp [h a s j g]<br>[w w z z c c d d e e p p b b h h a a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[v q y r] &nbsp [v q y l]<br>[t x f x u x q x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[n f k t i] &nbsp [n t k f i]<br>[o o x x t t x x n n x x m m x x k k x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[k f i u] &nbsp [k f i u v]<br>[a b e c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[s j g x] &nbsp [a j g x]<br>[s x a x h x b x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split1": [
      {
        "prompt": "[z b h a] &nbsp [z b h s]<br>[f f x x k k x x t t x x n n x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[u u v q y r] &nbsp [u v q y r]<br>[s x g x o x n x k x i x v x y x l x l x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[b h a s] &nbsp [b h a j]<br>[v i k n o g s h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[t k f i i u] &nbsp [t k f i u]<br>[l l x x r r x x y y x x q q x x v v x x v v x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[d e z b] &nbsp [d e z h]<br>[14 14 13 13 12 12 11 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[g x o m] &nbsp [g x o n]<br>[22 22 0 0 23 23 0 0 24 24 0 0 25 25 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[m n t k] &nbsp [m n t k f]<br>[p p x x d d x x z z x x h h x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[s j g x] &nbsp [s j g o]<br>[14 13 12 11 10 9 8 7] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[a s j g] &nbsp [h s j g]<br>[20 20 0 0 21 21 0 0 22 22 0 0 23 23 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[o m n c k] &nbsp [o m n t k]<br>[c c x x d d x x e e x x z z x x b b x x h h x x a a x x s s x x g g x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split2": [
      {
        "prompt": "[a s j g] &nbsp [a s j x]<br>[6 0 8 0 10 0 12 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[c d d e z b] &nbsp [c d e z b]<br>[t t x x f f x x u u x x q q x x q q x x r r x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[n t k f] &nbsp [m t k f]<br>[20 0 22 0 24 0 26 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[d e z b] &nbsp [c e z b]<br>[19 0 21 0 23 0 25 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[v q y r] &nbsp [v q y l]<br>[r q u f t m x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[v q y r] &nbsp [v q y l]<br>[h x s x g x o x n x k x i x v x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[h a s j] &nbsp [h a s g]<br>[14 14 0 0 15 15 0 0 16 16 0 0 17 17 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[t m n o k] &nbsp [o m n t k]<br>[5 5 7 7 9 9 13 13 11 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[m n t k] &nbsp [m n t f]<br>[z z h h s s g g o o n n k k i i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[g x w m n] &nbsp [g x o m n]<br>[1 3 10 7 9 11 13 15 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split3": [
      {
        "prompt": "[j g x l m] &nbsp [j g x o m]<br>[11 11 18 18 17 17 16 16 15 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[i u v q q y] &nbsp [i u v q y]<br>[18 0 19 0 19 0 20 0 21 0 22 0 23 0 24 0 25 0 26 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[e z b h a a] &nbsp [e z b h a]<br>[19 19 21 21 23 23 25 25 25 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[j g x o] &nbsp [s g x o]<br>[i x f x k x t x n x m x o x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[g x o m] &nbsp [g x o n]<br>[r q u f t m x j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[o z b h a] &nbsp [e z b h a]<br>[1 0 3 0 5 0 8 0 9 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[i y v q u] &nbsp [i u v q y]<br>[r x u x q x f x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[f i u v] &nbsp [k i u v]<br>[q u f t m x j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[u v q y] &nbsp [u v q r]<br>[14 14 13 13 12 12 11 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[u v q y] &nbsp [u v q r]<br>[13 11 9 7] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split4": [
      {
        "prompt": "[s j g x] &nbsp [s j g x o]<br>[5 5 4 4 3 3 2 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[j g m o x] &nbsp [j g x o m]<br>[l l r r y y u u v v q q i i f f k k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[n t k i f] &nbsp [n t k f i]<br>[t t x x n n x x x x x x o o x x m m x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[b h a s] &nbsp [b h a s j]<br>[l y v i k n o g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[w p c d] &nbsp [w p c d e]<br>[2 2 4 4 6 6 8 8] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[w p c e d] &nbsp [w p c d e]<br>[l l r r y y k k v v u u i i f f q q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[b h a s j j] &nbsp [b h a s j]<br>[y y v v i i k k n n n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[t k f i] &nbsp [t k f u]<br>[r r x x y y x x q q x x v v x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[x o r n t] &nbsp [x o m n t]<br>[14 0 16 0 18 0 12 0 22 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[h a s j] &nbsp [h a s g]<br>[16 16 15 15 14 14 13 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split5": [
      {
        "prompt": "[u v q y] &nbsp [u v q r]<br>[a a x x s s x x j j x x g g x x x x x x o o x x m m x x n n x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[z b h a] &nbsp [z b h a s]<br>[l x y x v x i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[g x o m] &nbsp [j x o m]<br>[v i k n o g s h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[f i u v r] &nbsp [f i u v q]<br>[1 1 3 3 5 5 20 20 9 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[e z b h] &nbsp [e z b h a]<br>[25 24 23 22 21 20 19 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[w p p c d e] &nbsp [w p c d e]<br>[j j x x g g x x x x x x o o x x m m x x n n x x t t x x k k x x f f x x f f x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[h a s x g] &nbsp [h a s j g]<br>[16 0 15 0 14 0 13 0 24 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[v q y r] &nbsp [v q y r l]<br>[j j a a b b e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[c d e z] &nbsp [c d e b]<br>[22 22 0 0 23 23 0 0 24 24 0 0 25 25 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[g j x o m] &nbsp [j g x o m]<br>[h h z z d d s s g g o o n n k k i i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split6": [
      {
        "prompt": "[n t k f] &nbsp [m t k f]<br>[7 7 9 9 11 11 13 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[g x o m] &nbsp [g x o n]<br>[2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[t k k f i u] &nbsp [t k f i u]<br>[22 0 22 0 21 0 20 0 19 0 18 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[k f f i u v] &nbsp [k f i u v]<br>[o o o o g g s s h h z z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[s j g g x o] &nbsp [s j g x o]<br>[24 0 24 0 23 0 22 0 21 0 20 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[i u v q] &nbsp [i u v y]<br>[16 0 15 0 14 0 13 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[w p c d] &nbsp [w p c e]<br>[25 25 24 24 23 23 22 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[a s j g] &nbsp [a s j x]<br>[8 8 10 10 12 12 14 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[c d e z] &nbsp [p d e z]<br>[h h b b z z e e d d c c p p w w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[j g x o m m] &nbsp [j g x o m]<br>[v x u x i x i x f x k x t x n x m x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gensplit7": [
      {
        "prompt": "[j g x o] &nbsp [j g x m]<br>[11 0 10 0 9 0 8 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[i k f t u] &nbsp [t k f i u]<br>[f f x x n n x x t t x x k k x x m m x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[v q y r] &nbsp [v q y l]<br>[4 6 8 10 12 14 16 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[g x o m] &nbsp [g x o n]<br>[16 16 18 18 20 20 22 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[u v q y] &nbsp [u v q r]<br>[c x e x b x a x j x x x m x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[e z b h] &nbsp [e z b h a]<br>[14 12 10 8] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[x o m n] &nbsp [g o m n]<br>[4 4 3 3 2 2 1 1] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[w p c d] &nbsp [w p c d e]<br>[n n m m o o x x g g j j s s a a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[l b h a s] &nbsp [z b h a s]<br>[g g x x x x x x o o x x l l x x n n x x t t x x k k x x f f x x i i x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[u v q y] &nbsp [i v q y]<br>[z z x x e e x x d d x x c c x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ]
  },
  "alph_5": {
    "shuffled_letters": [
      "r",
      "y",
      "t",
      "b",
      "n",
      "x",
      "s",
      "z",
      "p",
      "d",
      "u",
      "v",
      "o",
      "g",
      "i",
      "k",
      "h",
      "l",
      "w",
      "c"
    ],
    "shuffled_alphabet": [
      "a",
      "r",
      "y",
      "t",
      "e",
      "f",
      "b",
      "n",
      "x",
      "j",
      "s",
      "z",
      "m",
      "p",
      "d",
      "u",
      "q",
      "v",
      "o",
      "g",
      "i",
      "k",
      "h",
      "l",
      "w",
      "c"
    ],
    "larger_int": [
      {
        "prompt": "[t e b f n] &nbsp [t e f b n]<br>[d i o q h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[u q v o g g] &nbsp [u q v o g]<br>[p u u v g k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[r y t e] &nbsp [r y t f]<br>[e b x s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[k h l w] &nbsp [k h l c]<br>[m d q o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[z m p d] &nbsp [z m p u]<br>[m d q o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[t e n b f] &nbsp [t e f b n]<br>[v l k g c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[n x j s] &nbsp [n x j z]<br>[t f n j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[r y t e] &nbsp [r y t f]<br>[f n j z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[g i k h] &nbsp [g i k l]<br>[q o i h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[i k h l w w] &nbsp [i k h l w]<br>[q o i h h w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "longer_targ": [
      {
        "prompt": "[y t e f] &nbsp [r t e f]<br>[r y t e f b n x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[g o i k h] &nbsp [o g i k h]<br>[b x n j s z m p d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[g i k h] &nbsp [o i k h]<br>[f b n x j s z m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[y t e f] &nbsp [r t e f]<br>[u q v o g i k h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[m d p u q] &nbsp [m p d u q]<br>[q d u p v o g i k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[n x j s] &nbsp [n x j z]<br>[j s z m p d u q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[s z m p] &nbsp [s z m d]<br>[f b n x j s z m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[m p d u u q] &nbsp [m p d u q]<br>[x j s z m m p d u q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[p d u q] &nbsp [p d u v]<br>[u q v o g i k h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[b n x j] &nbsp [b n x j s]<br>[p d u q v o g i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "group": [
      {
        "prompt": "[f b n x] &nbsp [f b n j]<br>[e e f f b b n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[z m p d] &nbsp [z m p u]<br>[v v o o g g i i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[m z p d u] &nbsp [z m p d u]<br>[n n s s j j x x z z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[o g i b h] &nbsp [o g i k h]<br>[e e o o b b n n x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[s t m p d] &nbsp [s z m p d]<br>[u u w w v v o o g g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[e f b n] &nbsp [e f b x]<br>[k k h h l l w w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[f b n x] &nbsp [e b n x]<br>[s s z z m m p p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[g i k h] &nbsp [g i k h l]<br>[p p d d u u q q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[b b n x j s] &nbsp [b n x j s]<br>[k k h h l l w w w w c c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[o g i k] &nbsp [o g i h]<br>[b b n n x x j j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "interleaved": [
      {
        "prompt": "[b n x j] &nbsp [b n x s]<br>[g x i x k x h x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[s z m p] &nbsp [s z m p d]<br>[i x k x h x l x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[o g i k] &nbsp [o g i h]<br>[n x x x j x s x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[i k h l] &nbsp [i k h l w]<br>[a x r x y x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[w h l k c] &nbsp [k h l w c]<br>[q x g x o x v x i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[d u q v] &nbsp [p u q v]<br>[p x d x u x q x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[y t e f] &nbsp [y t e b]<br>[u x q x v x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[o g i k] &nbsp [o g i h]<br>[k x h x l x w x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[v o g i] &nbsp [v o g k]<br>[e x f x b x n x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[b n x j] &nbsp [b n x s]<br>[a x r x y x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "letter2num": [
      {
        "prompt": "[v o g i] &nbsp [v o g k]<br>[19 20 21 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[z m p d] &nbsp [s m p d]<br>[21 22 23 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[o g i k] &nbsp [o g i h]<br>[12 13 14 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[v v o g i k] &nbsp [v o g i k]<br>[16 17 18 18 19 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[b n x j] &nbsp [f n x j]<br>[10 11 12 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[b n x j] &nbsp [f n x j]<br>[8 9 10 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[k h o w c] &nbsp [k h l w c]<br>[9 10 11 8 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[s z m p r] &nbsp [s z m p d]<br>[16 17 18 19 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[o v q g i] &nbsp [q v o g i]<br>[11 15 13 14 12] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[k h l w w c] &nbsp [k h l w c]<br>[20 21 22 23 24 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "reverse": [
      {
        "prompt": "[o g i k] &nbsp [o g i h]<br>[l h k i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[a r y t t e] &nbsp [a r y t e]<br>[x n b f f e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[t e f b] &nbsp [t e f n]<br>[j x n b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[s z m p] &nbsp [s z m p d]<br>[j x n b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[t e f b] &nbsp [t e f n]<br>[h k i g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[d u q v] &nbsp [d u q v o]<br>[w l h k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[a r y t] &nbsp [a r y e]<br>[j x n b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[s z m p] &nbsp [s z m p d]<br>[l h k i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[e f b n] &nbsp [e f b x]<br>[i g o v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[k g l w c] &nbsp [k h l w c]<br>[p m a s j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split1": [
      {
        "prompt": "[s m z p d] &nbsp [s z m p d]<br>[13 11 12 10 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[e f b n] &nbsp [e f b n x]<br>[j n f t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[d u q v] &nbsp [p u q v]<br>[t x f x n x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[h l w c] &nbsp [k l w c]<br>[16 17 18 19 20 21 22 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[k h l w] &nbsp [k h l c]<br>[2 4 6 8] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[k h l w] &nbsp [k h l c]<br>[u u x x q q x x v v x x o o x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[m p d u] &nbsp [m p d q]<br>[y x e x b x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[o g k i h] &nbsp [o g i k h]<br>[d x u x q x i x o x g x v x k x h x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[i k h h l w] &nbsp [i k h l w]<br>[5 4 4 3 2 1] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[e f b n] &nbsp [e f b x]<br>[18 17 16 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split2": [
      {
        "prompt": "[d d u q v o] &nbsp [d u q v o]<br>[e e f f b b n n x x j j s s z z m m m m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[r y t e e f] &nbsp [r y t e f]<br>[t f n j z p u u v g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[e y t r f] &nbsp [r y t e f]<br>[7 13 11 9 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[p u d q v] &nbsp [p d u q v]<br>[21 20 19 17 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[k k h l w c] &nbsp [k h l w c]<br>[19 0 20 0 21 0 21 0 22 0 23 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[v o g i] &nbsp [v o g i k]<br>[17 19 21 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[c h l w k] &nbsp [k h l w c]<br>[17 21 19 23 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[o h l w c] &nbsp [k h l w c]<br>[11 12 13 14 15 16 17 18 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[a a r y t e] &nbsp [a r y t e]<br>[m x p x d x u x q x v x v x o x g x i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[p d u q q v] &nbsp [p d u q v]<br>[g g g g o o v v q q u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split3": [
      {
        "prompt": "[g i k h] &nbsp [g i k l]<br>[18 0 19 0 20 0 21 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[i k r l w] &nbsp [i k h l w]<br>[23 23 13 13 14 14 15 15 16 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[p d u q] &nbsp [p d u q v]<br>[y y t t e e f f b b n n x x j j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[t e f b] &nbsp [t e f n]<br>[19 19 20 20 21 21 22 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[p p d u q v] &nbsp [p d u q v]<br>[v v o o o o g g i i k k h h l l w w c c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[i k y l w] &nbsp [i k h l w]<br>[a x r x y x t x e x f x l x n x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[g i k h] &nbsp [o i k h]<br>[o o g g i i k k h h l l w w c c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[y t e f] &nbsp [y t e b]<br>[w h i o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[d u o v q] &nbsp [d u q v o]<br>[24 25 26 23 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[g k i h l] &nbsp [g i k h l]<br>[f f x x e e x x b b x x n n x x x x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split4": [
      {
        "prompt": "[e f b n] &nbsp [e f b x]<br>[y y e e b b x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[b n x j j s] &nbsp [b n x j s]<br>[a x a x r x y x t x e x f x b x n x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[z m p d] &nbsp [z m p u]<br>[k k x x h h x x l l x x w w x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[x j s z] &nbsp [x j s z m]<br>[w x l x h x k x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[k h l w] &nbsp [i h l w]<br>[5 7 9 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[k h h l w c] &nbsp [k h l w c]<br>[a x r x y x t x e x f x b x n x n x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[i k w l h] &nbsp [i k h l w]<br>[v x o x h x i x k x g x l x w x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[k h l w] &nbsp [k h l c]<br>[17 18 19 20 21 22 23 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[o g i k] &nbsp [v g i k]<br>[9 11 13 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[m p d u] &nbsp [m p d q]<br>[v v o o g g i i k k h h l l w w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split5": [
      {
        "prompt": "[x j s z] &nbsp [x j s m]<br>[s x b e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[y t e f b b] &nbsp [y t e f b]<br>[f n n j z p u v g k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[n b f x j] &nbsp [f b n x j]<br>[u v g p z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[v o g i] &nbsp [v o g k]<br>[15 16 17 18 19 20 21 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[g i k h] &nbsp [g i k l]<br>[q q o o i i h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[y t e e f b] &nbsp [y t e f b]<br>[i g o v q u d p m m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[q v v o g i] &nbsp [q v o g i]<br>[24 23 22 22 21 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[v o g i] &nbsp [v o g k]<br>[17 19 21 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[x r y t e] &nbsp [a r y t e]<br>[12 13 14 15 16 17 22 19 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[m e f b n] &nbsp [t e f b n]<br>[k g i u p] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split6": [
      {
        "prompt": "[x j s z q] &nbsp [x j s z m]<br>[8 10 12 20 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[z e f b n] &nbsp [t e f b n]<br>[m x h x k x i x g x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[g q v o u] &nbsp [u q v o g]<br>[2 2 3 3 5 5 4 4 6 6] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[l h k w c] &nbsp [k h l w c]<br>[q q g g o o v v i i k k h h l l w w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[k h l w] &nbsp [k h l c]<br>[21 21 22 22 23 23 24 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[o g i k] &nbsp [v g i k]<br>[7 7 8 8 9 9 10 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[r y t e] &nbsp [a y t e]<br>[22 22 23 23 24 24 25 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[z m p d d u] &nbsp [z m p d u]<br>[r r t f n j z p u v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[j s z m] &nbsp [j s z p]<br>[h k i g o v q u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[v o g i] &nbsp [v o g i k]<br>[x n b f e t y r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gensplit7": [
      {
        "prompt": "[k k h l w c] &nbsp [k h l w c]<br>[j j x x n n n n b b f f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[u q v o] &nbsp [d q v o]<br>[12 12 13 13 14 14 15 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[r y t e] &nbsp [r y t e f]<br>[y e b x s m d q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[x j s z] &nbsp [x j s m]<br>[m s x b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[i k h l p] &nbsp [i k h l w]<br>[5 23 22 21 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[k h l w] &nbsp [k h l c]<br>[i x g x o x v x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[j s s z m p] &nbsp [j s z m p]<br>[n n j j z z p p p p u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[n x j s] &nbsp [b x j s]<br>[n n b b f f e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[f b n x] &nbsp [f b n x j]<br>[b b x x n n x x x x x x j j x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[b n x j] &nbsp [f n x j]<br>[i o q d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split1": [
      {
        "prompt": "[v a g i k] &nbsp [v o g i k]<br>[v x q x k x d x p x m x z x s x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[z m p d d u] &nbsp [z m p d u]<br>[w h i o q q d m s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[b n x j] &nbsp [f n x j]<br>[25 0 24 0 23 0 22 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[t e f b] &nbsp [t e f b n]<br>[23 0 22 0 21 0 20 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[b n x j] &nbsp [b n x s]<br>[v v x x o o x x g g x x i i x x k k x x h h x x l l x x w w x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[p d u r v] &nbsp [p d u q v]<br>[3 3 5 5 7 7 9 9 22 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[e f b n] &nbsp [e f b x]<br>[h h x x k k x x i i x x g g x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[v o g k i] &nbsp [v o g i k]<br>[6 6 5 5 4 4 2 2 3 3] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[n x j z s] &nbsp [n x j s z]<br>[w h i o q d x s m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[i k h h l w] &nbsp [i k h l w]<br>[10 8 6 6 4 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split2": [
      {
        "prompt": "[r y t e] &nbsp [a y t e]<br>[u u x x v v x x g g x x k k x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[v o g i] &nbsp [q o g i]<br>[e e x x b b x x x x x x s s x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[i k h l w w] &nbsp [i k h l w]<br>[c x w x l x h x k x i x g x o x o x v x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[k h l w] &nbsp [i h l w]<br>[5 5 7 7 9 9 11 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[j s z m] &nbsp [j s z m p]<br>[13 13 15 15 17 17 19 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[m p d u] &nbsp [m p d q]<br>[14 0 15 0 16 0 17 0 18 0 19 0 20 0 21 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[i k h l] &nbsp [i k h l w]<br>[1 3 5 7 9 11 13 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[i f b n x] &nbsp [e f b n x]<br>[y x e x b x x x s x m x h x q x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[k h l w] &nbsp [i h l w]<br>[10 0 11 0 12 0 13 0 14 0 15 0 16 0 17 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[b n x j] &nbsp [b n x j s]<br>[f f n n j j z z p p u u v v g g] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split3": [
      {
        "prompt": "[i w h l k] &nbsp [i k h l w]<br>[v v x x z z x x p p x x u u x x j j x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[g i k h h l] &nbsp [g i k h l]<br>[d x m x s x s x x x b x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[y t e f] &nbsp [r t e f]<br>[11 11 10 10 9 9 8 8] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[a r y t] &nbsp [a r y e]<br>[19 19 0 0 20 20 0 0 21 21 0 0 22 22 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[u q v o] &nbsp [u q v g]<br>[7 7 8 8 9 9 10 10 11 11 12 12 13 13 14 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[q v o g] &nbsp [u v o g]<br>[25 24 23 22 21 20 19 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[t e f b] &nbsp [t e f n]<br>[k x i x g x o x v x q x u x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[d s z m p] &nbsp [j s z m p]<br>[15 17 19 14 23 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[k h a w c] &nbsp [k h l w c]<br>[u u x x m m x x g g x x k k x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[n x j s] &nbsp [n x j z]<br>[26 25 24 23 22 21 20 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split4": [
      {
        "prompt": "[p d u q] &nbsp [p d u v]<br>[9 9 0 0 10 10 0 0 11 11 0 0 12 12 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[d u q v] &nbsp [d u q o]<br>[v v x x g g x x k k x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[k h l w w c] &nbsp [k h l w c]<br>[b x x x s x m x m x d x q x o x i x h x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[y t e f] &nbsp [y t e b]<br>[n x j x z x p x u x v x g x k x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[r y t e] &nbsp [r y t f]<br>[g x v x u x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[x j s z] &nbsp [x j s z m]<br>[23 23 22 22 21 21 20 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[g i f h l] &nbsp [g i k h l]<br>[13 13 16 16 18 18 20 20 22 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[n f b e x] &nbsp [e f b n x]<br>[c l g k v u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[i k h l] &nbsp [i k h l w]<br>[m x z x s x j x x x n x b x f x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[k h l w] &nbsp [k h l c]<br>[16 15 14 13 12 11 10 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split5": [
      {
        "prompt": "[k h l w] &nbsp [k h l c]<br>[11 0 12 0 13 0 14 0 15 0 16 0 17 0 18 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[x n j s z] &nbsp [n x j s z]<br>[10 12 14 16 18 22 20 24 26] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[i k h c w] &nbsp [i k h l w]<br>[26 25 24 23 22 13 20 19 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[a r y t] &nbsp [a r y t e]<br>[13 12 11 10 9 8 7 6] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[j l z m p] &nbsp [j s z m p]<br>[17 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 9 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[o g i k] &nbsp [v g i k]<br>[l k g v u p z j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[s z z m p d] &nbsp [s z m p d]<br>[2 4 4 6 8 10 12 14 16 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[t e f b] &nbsp [t e f n]<br>[q q x x o o x x i i x x h h x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[x j s t m] &nbsp [x j s z m]<br>[c x w x l x h x n x i x g x o x v x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[e f b n] &nbsp [t f b n]<br>[n n x x b b x x f f x x e e x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split6": [
      {
        "prompt": "[a r y t] &nbsp [a r y e]<br>[q q u u d d p p m m z z s s j j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[j s z m] &nbsp [j s z m p]<br>[9 0 10 0 11 0 12 0 13 0 14 0 15 0 16 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[m e d u q] &nbsp [m p d u q]<br>[z z x x n n f f t t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[q v o g] &nbsp [q v o i]<br>[r r t t f f n n j j z z p p u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[z m p d] &nbsp [s m p d]<br>[d d m m s s x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[n x j s] &nbsp [n x j z]<br>[z z x x p p x x u u x x v v x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[d u q v] &nbsp [d u q o]<br>[v x u x p x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[y t e f] &nbsp [y t e f b]<br>[c x l x k x g x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[p e f b n] &nbsp [t e f b n]<br>[x x s x m x d x p x o x i x h x w x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[y t e j b] &nbsp [y t e f b]<br>[j x z x p x u x v x g x b x l x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gensplit7": [
      {
        "prompt": "[x j s s z m] &nbsp [x j s z m]<br>[j x z x p x u x v x v x g x k x l x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[a r y t] &nbsp [a r y e]<br>[10 12 14 16 18 20 22 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[o g g i k h] &nbsp [o g i k h]<br>[s s x x z z x x z z x x m m x x p p x x d d x x u u x x q q x x v v x x o o x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[r y t e] &nbsp [r y t f]<br>[1 0 3 0 5 0 7 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[q v o g] &nbsp [u v o g]<br>[s x j x x x n x b x f x e x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[o g i k] &nbsp [o g i k h]<br>[16 18 20 22 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[b n x j] &nbsp [b n x s]<br>[c l k g v u p z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[y t e f] &nbsp [y t e b]<br>[18 0 20 0 22 0 24 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[k h l w] &nbsp [k h l c]<br>[9 8 7 6 5 4 3 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[d u q v] &nbsp [d u q o]<br>[p p z z j j n n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ]
  },
  "alph_6": {
    "shuffled_letters": [
      "x",
      "p",
      "j",
      "h",
      "n",
      "a",
      "u",
      "d",
      "q",
      "c",
      "t",
      "f",
      "s",
      "b",
      "g",
      "y",
      "e",
      "z",
      "l",
      "o"
    ],
    "shuffled_alphabet": [
      "x",
      "p",
      "j",
      "h",
      "n",
      "a",
      "u",
      "d",
      "i",
      "q",
      "k",
      "c",
      "m",
      "t",
      "f",
      "s",
      "b",
      "r",
      "g",
      "y",
      "e",
      "v",
      "w",
      "z",
      "l",
      "o"
    ],
    "larger_int": [
      {
        "prompt": "[x p j h h n] &nbsp [x p j h n]<br>[f b g e e w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[y r g b e] &nbsp [b r g y e]<br>[k i m f b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[q c k m t] &nbsp [q k c m t]<br>[k m g b f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[d i q k] &nbsp [u i q k]<br>[s r y v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[q k c m] &nbsp [q k c t]<br>[d q c t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[w z l o] &nbsp [v z l o]<br>[b g e w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[i q k c] &nbsp [i q k c m]<br>[b g e w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[f s b r] &nbsp [f s b r g]<br>[r y v z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[v w z l] &nbsp [v w z l o]<br>[x j n u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[w v z l o] &nbsp [v w z l o]<br>[g e l w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "longer_targ": [
      {
        "prompt": "[f s b r] &nbsp [f s b g]<br>[t f s b r g y e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[q k c m] &nbsp [q k c t]<br>[j h n a u d i q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[k b r g y] &nbsp [s b r g y]<br>[b r p y e v w z l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[t f s b] &nbsp [t f s r]<br>[r g y e v w z l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[v w z l] &nbsp [v w z o]<br>[s b r g y e v w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[n a u d] &nbsp [n a u d i]<br>[a u d i q k c m] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[a u d i] &nbsp [a u d i q]<br>[r g y e v w z l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[d i q k] &nbsp [d i q c]<br>[u d i q k c m t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[g y e v w w] &nbsp [g y e v w]<br>[i q k c m t f f s b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[k k c m t f] &nbsp [k c m t f]<br>[b r g y e v w w z l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "group": [
      {
        "prompt": "[e v w z] &nbsp [e v w z l]<br>[c c m m t t f f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[p j h n] &nbsp [p j h n a]<br>[i i q q k k c c] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[x p j h] &nbsp [x p j n]<br>[a a u u d d i i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[p j h n g] &nbsp [p j h n a]<br>[c c m m t t r r s s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[v w z l] &nbsp [v w z o]<br>[g g y y e e v v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[q k c m] &nbsp [q k c t]<br>[m m t t f f s s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[d i f k c] &nbsp [d i q k c]<br>[j j m m n n a a u u] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[n a a u d i] &nbsp [n a u d i]<br>[p p j j h h h h n n a a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[j h n a u u] &nbsp [j h n a u]<br>[b b b b r r g g y y e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[s b r g] &nbsp [s b r y]<br>[c c m m t t f f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "interleaved": [
      {
        "prompt": "[a u d i] &nbsp [a u d q]<br>[y x e x v x w x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[x p j h] &nbsp [x p j n]<br>[n x a x u x d x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[k k c m t f] &nbsp [k c m t f]<br>[y x y x e x v x w x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[a u d i] &nbsp [n u d i]<br>[u x d x i x q x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[c y e v w] &nbsp [g y e v w]<br>[n x j x u x d x i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[m t f s] &nbsp [m t f s b]<br>[x x p x j x h x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[t f s b] &nbsp [m f s b]<br>[j x h x n x a x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[p p j h n a] &nbsp [p j h n a]<br>[x x p x j x h x n x n x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[c m t f] &nbsp [c m t s]<br>[x x p x j x h x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[q k c m] &nbsp [q k c t]<br>[h x n x a x u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "letter2num": [
      {
        "prompt": "[n a f d i] &nbsp [n a u d i]<br>[1 18 19 20 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[x p j h] &nbsp [x p j n]<br>[21 22 23 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[h n a u] &nbsp [h n a d]<br>[14 15 16 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[u d i i q k] &nbsp [u d i q k]<br>[21 22 23 23 24 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[v w z l o o] &nbsp [v w z l o]<br>[17 17 18 19 20 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[b r g y] &nbsp [b r g e]<br>[10 11 12 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[v w z l] &nbsp [v w z l o]<br>[21 22 23 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[k c m t] &nbsp [k c m f]<br>[4 5 6 7] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[n a a u d i] &nbsp [n a u d i]<br>[20 20 21 22 23 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[d i q k] &nbsp [d i q k c]<br>[15 16 17 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "reverse": [
      {
        "prompt": "[r g y e] &nbsp [r g y v]<br>[l z w v] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[x p j h h n] &nbsp [x p j h n]<br>[d u u a n h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[m t f s] &nbsp [m t f b]<br>[g r b s] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[f j h n a] &nbsp [p j h n a]<br>[o z w v e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[i q k c] &nbsp [i q k m]<br>[z w v e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[g y e v] &nbsp [g y e w]<br>[o l z w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[k c c m t f] &nbsp [k c m t f]<br>[r b b s f t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[n a u d] &nbsp [h a u d]<br>[k q i d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[f s b r g g] &nbsp [f s b r g]<br>[e y g r b b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[v w z l] &nbsp [e w z l]<br>[a n h j] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split1": [
      {
        "prompt": "[k c m t] &nbsp [q c m t]<br>[f x b x g x e x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[x p j h] &nbsp [x p j n]<br>[u u a a n n h h] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[f s b r] &nbsp [f s b g]<br>[5 4 3 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[x p j h] &nbsp [x p j n]<br>[y r s t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[d i q k] &nbsp [u i q k]<br>[r x g x y x e x v x w x z x l x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[b r g y] &nbsp [s r g y]<br>[18 19 20 21 22 23 24 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[n a u d] &nbsp [h a u d]<br>[e e y y g g r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[d a u n i] &nbsp [n a u d i]<br>[v v x x e e x x w w x x z z x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[r g y e] &nbsp [b g y e]<br>[f f b b g g e e] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[u a d i q] &nbsp [a u d i q]<br>[z z x x w w x x v v x x l l x x o o x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split2": [
      {
        "prompt": "[t f s b] &nbsp [t f s r]<br>[6 6 7 7 8 8 9 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[e v w w z l] &nbsp [e v w z l]<br>[d d x x i i x x q q x x k k x x c c x x c c x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[n a u d] &nbsp [h a u d]<br>[18 17 16 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[p j n h a] &nbsp [p j h n a]<br>[13 0 17 0 15 0 16 0 14 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[p j h n] &nbsp [x j h n]<br>[l z w v e y g r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[t f s b] &nbsp [m f s b]<br>[a x d x q x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[j h n a] &nbsp [j h n a u]<br>[11 11 12 12 13 13 14 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[j h n a] &nbsp [j h n u]<br>[a x n x h x j x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[r g z e v] &nbsp [r g y e v]<br>[9 11 16 15 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[b r g y] &nbsp [b r g e]<br>[r x y x v x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split3": [
      {
        "prompt": "[t f s b r r] &nbsp [t f s b r]<br>[q c t s r y v z z o] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[b r g y] &nbsp [b r g e]<br>[m c k q i d u a] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[j v n a u] &nbsp [j h n a u]<br>[5 4 20 2 1] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[t f s b] &nbsp [t f s b r]<br>[25 24 23 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[m t f s y] &nbsp [m t f s b]<br>[20 0 21 0 22 0 12 0 24 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[e v z w l] &nbsp [e v w z l]<br>[n x h x p x j x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[h a n u d] &nbsp [h n a u d]<br>[12 20 14 15 16 17 18 19 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[s b r g] &nbsp [f b r g]<br>[m m x x t t x x f f x x s s x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[s b r g] &nbsp [s b r g y]<br>[x x p p j j h h n n a a u u d d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[a u d i] &nbsp [a u d i q]<br>[c c t t s s r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split4": [
      {
        "prompt": "[t r s b f] &nbsp [t f s b r]<br>[t t m m c c f f s s b b r r g g y y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[i q k c] &nbsp [i q k m]<br>[13 14 15 16 17 18 19 20] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[m t f s] &nbsp [m t f b]<br>[c x t x s x r x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[u q i d k] &nbsp [u d i q k]<br>[q x c x r x s x t x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[e v w z] &nbsp [e v w z l]<br>[9 10 11 12 13 14 15 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[q k c m] &nbsp [q k c m t]<br>[y r s t] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[e v w z] &nbsp [e v w z l]<br>[10 10 11 11 12 12 13 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[y e v w] &nbsp [y e v w z]<br>[o o l l z z w w] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[f s b r] &nbsp [t s b r]<br>[12 0 13 0 14 0 15 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[x p j h] &nbsp [x p j n]<br>[h x a x d x q x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split5": [
      {
        "prompt": "[a u d i] &nbsp [n u d i]<br>[k x m x f x b x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[n a u d i i] &nbsp [n a u d i]<br>[r x g x y x e x e x v x w x z x l x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[n a u d] &nbsp [h a u d]<br>[21 23 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[v w z l l o] &nbsp [v w z l o]<br>[x x j x n x n x u x i x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[v w z l] &nbsp [v w z o]<br>[5 7 9 11] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[s b r g] &nbsp [f b r g]<br>[13 12 11 10] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[k c t m f] &nbsp [k c m t f]<br>[2 0 4 0 3 0 5 0 6 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[o a u d i] &nbsp [n a u d i]<br>[7 0 21 0 22 0 23 0 24 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[d i q k] &nbsp [d i q c]<br>[e g b f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[w v e z l] &nbsp [e v w z l]<br>[6 6 7 7 10 10 9 9 8 8] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gen_split6": [
      {
        "prompt": "[b r g y] &nbsp [b r g e]<br>[s x f x t x m x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[n a u d] &nbsp [n a u d i]<br>[19 19 20 20 21 21 22 22] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[i q k c] &nbsp [i q k c m]<br>[c k q i d u a n] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[v w z l] &nbsp [v w z o]<br>[10 11 12 13 14 15 16 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[m t f f s b] &nbsp [m t f s b]<br>[19 0 20 0 21 0 22 0 22 0 23 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[i q k c] &nbsp [i q k c m]<br>[r r g g y y e e v v w w z z l l] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[x p v h n] &nbsp [x p j h n]<br>[1 21 23 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[n a u d] &nbsp [n a u i]<br>[e g b f] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[u d i q] &nbsp [a d i q]<br>[12 14 16 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[w e v y z] &nbsp [y e v w z]<br>[12 0 9 0 10 0 11 0 8 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "2gensplit7": [
      {
        "prompt": "[x p j h n n] &nbsp [x p j h n]<br>[d x q x q x c x t x s x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[x p j h] &nbsp [x p j h n]<br>[8 0 9 0 10 0 11 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[d q i k c] &nbsp [d i q k c]<br>[3 2 1 4 5 6 7 8 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[r s y e v] &nbsp [r g y e v]<br>[1 2 3 4 5 6 13 8 9] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[n a u u d i] &nbsp [n a u d i]<br>[z z w w v v v v e e y y] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[i q k c] &nbsp [d q k c]<br>[13 13 14 14 15 15 16 16] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[a u d i] &nbsp [n u d i]<br>[q c t s r y v z] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[x p j h] &nbsp [x p j n]<br>[r x y x v x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[r f s b t] &nbsp [t f s b r]<br>[i j n u x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[y e v w] &nbsp [y e v w z]<br>[i x k x m x f x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split1": [
      {
        "prompt": "[x p h j n] &nbsp [x p j h n]<br>[v v y y t t s s r r] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[c m t f] &nbsp [c m t s]<br>[10 0 12 0 14 0 16 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[g y e v] &nbsp [g y e w]<br>[h x a x d x q x c x t x s x r x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[t f s b] &nbsp [t f s r]<br>[15 0 16 0 17 0 18 0 19 0 20 0 21 0 22 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[x p j h] &nbsp [x p j n]<br>[v v x x e e x x y y x x g g x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[x p h j n] &nbsp [x p j h n]<br>[6 6 4 4 5 5 3 3 2 2] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[f s b r] &nbsp [f s b r g]<br>[25 0 24 0 23 0 22 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[x p j h] &nbsp [x p j h n]<br>[u u x x d d x x i i x x q q x x k k x x c c x x m m x x t t x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[f s b r] &nbsp [t s b r]<br>[8 7 6 5 4 3 2 1] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[j h n a] &nbsp [j h n u]<br>[y x g x r x b x s x f x t x m x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split2": [
      {
        "prompt": "[g y e v] &nbsp [g y e w]<br>[c x q x d x a x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[u d i q] &nbsp [u d i k]<br>[26 25 24 23 22 21 20 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[a u d d i q] &nbsp [a u d i q]<br>[r r x x b b x x b b x x s s x x f f x x t t x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[k c m t] &nbsp [k c m t f]<br>[q q x x c c x x t t x x s s x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[n a u d d i] &nbsp [n a u d i]<br>[p p x x j j x x h h x x n n x x a a x x u u x x d d x x d d x x i i x x q q x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[y e v w] &nbsp [g e v w]<br>[20 20 19 19 18 18 17 17] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[b r g y] &nbsp [s r g y]<br>[y x g x r x b x s x f x t x m x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[a u d i] &nbsp [a u d i q]<br>[9 11 13 15 17 19 21 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[e v w z] &nbsp [e v w l]<br>[v y r s t c q d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[f s b r] &nbsp [f s b g]<br>[7 7 8 8 9 9 10 10 11 11 12 12 13 13 14 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split3": [
      {
        "prompt": "[v w z a o] &nbsp [v w z l o]<br>[6 0 8 0 10 0 12 0 2 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[u d i q] &nbsp [u d i k]<br>[s x t x c x q x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[y e v v w z] &nbsp [y e v w z]<br>[16 16 18 18 20 20 22 22 22 22 24 24] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[i q k m c] &nbsp [i q k c m]<br>[f f x x s s x x g g x x r r x x b b x x y y x x e e x x v v x x w w x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[a a u d i q] &nbsp [a u d i q]<br>[q x i x d x u x a x a x n x h x j x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[k c m t] &nbsp [q c m t]<br>[13 13 15 15 17 17 19 19] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[p j h n t] &nbsp [p j h n a]<br>[15 0 16 0 17 0 18 0 19 0 20 0 21 0 22 0 10 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[y e v w] &nbsp [y e v z]<br>[16 0 17 0 18 0 19 0 20 0 21 0 22 0 23 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[q k c m] &nbsp [q k c t]<br>[16 0 17 0 18 0 19 0 20 0 21 0 22 0 23 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[j a n h u] &nbsp [j h n a u]<br>[v x c x t x s x r x y x q x z x o x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split4": [
      {
        "prompt": "[t n a u d] &nbsp [h n a u d]<br>[r r s s t t k k q q] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[s b r g] &nbsp [s b r g y]<br>[21 21 20 20 19 19 18 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[k c m t] &nbsp [k c m t f]<br>[20 20 0 0 21 21 0 0 22 22 0 0 23 23 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[a i d u q] &nbsp [a u d i q]<br>[9 8 2 6 5 4 3 7 1] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[n a u d] &nbsp [n a u i]<br>[20 20 0 0 21 21 0 0 22 22 0 0 23 23 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[t m c f s] &nbsp [c m t f s]<br>[5 7 9 11 13 17 15 19 21] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[s b r g] &nbsp [s b r y]<br>[k x q x i x d x u x a x n x h x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[g g y e v w] &nbsp [g y e v w]<br>[2 0 4 0 6 0 6 0 8 0 10 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[y e v w] &nbsp [g e v w]<br>[19 0 21 0 23 0 25 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[r g y e e v] &nbsp [r g y e v]<br>[b b f f m m m m k k i i] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split5": [
      {
        "prompt": "[c m t f] &nbsp [k m t f]<br>[h h x x a a x x d d x x q q x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[i q k c] &nbsp [i q k c m]<br>[n n x x h h x x j j x x p p x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[v w z l] &nbsp [e w z l]<br>[w w e e g g b b] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[t f s b] &nbsp [t f s r]<br>[q x c x t x s x r x y x v x z x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[a u d i] &nbsp [a u d q]<br>[b b x x g g x x e e x x w w x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[d i q k] &nbsp [d i q c]<br>[16 16 17 17 18 18 19 19 20 20 21 21 22 22 23 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[c t m f s] &nbsp [c m t f s]<br>[r r x x v v x x y y x x e e x x g g x x w w x x z z x x l l x x o o x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[b r g y] &nbsp [s r g y]<br>[g x r x b x s x f x t x m x c x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[b s g y e] &nbsp [b r g y e]<br>[7 7 24 24 11 11 13 13 15 15] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[c m t f s s] &nbsp [c m t f s]<br>[i x u x n x j x j x x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gen_split6": [
      {
        "prompt": "[f s b r] &nbsp [f s b r g]<br>[19 19 21 21 23 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[h n a u u d] &nbsp [h n a u d]<br>[f f x x s s x x b b x x r r x x g g x x y y x x e e x x e e x x v v x x w w x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[h i q k c] &nbsp [d i q k c]<br>[q x i x r x u x a x n x h x j x p x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[c m t f] &nbsp [c m t s]<br>[11 11 0 0 12 12 0 0 13 13 0 0 14 14 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[v w z l o o] &nbsp [v w z l o]<br>[h h x x n n x x a a x x u u x x d d x x i i x x q q x x k k x x c c x x c c x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[b r g y e e] &nbsp [b r g y e]<br>[19 19 19 19 21 21 23 23 25 25] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[t t f s b r] &nbsp [t f s b r]<br>[g g g g b b f f m m k k] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[e v w z] &nbsp [y v w z]<br>[f f t t m m c c k k q q i i d d] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[v w z l o o] &nbsp [v w z l o]<br>[18 0 18 0 20 0 22 0 24 0 26 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[r g y e] &nbsp [r g y v]<br>[15 0 16 0 17 0 18 0 19 0 20 0 21 0 22 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ],
    "3gensplit7": [
      {
        "prompt": "[k c m t] &nbsp [k c m f]<br>[9 11 13 15 17 19 21 23] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 0
      },
      {
        "prompt": "[h n a u] &nbsp [h n a u d]<br>[r r x x g g x x y y x x e e x x v v x x w w x x z z x x l l x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 1
      },
      {
        "prompt": "[n a u d] &nbsp [n a u d i]<br>[j x n x u x i x k x m x f x b x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 2
      },
      {
        "prompt": "[p j h n] &nbsp [x j h n]<br>[8 8 10 10 12 12 14 14] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 3
      },
      {
        "prompt": "[k c c m t f] &nbsp [k c m t f]<br>[21 21 19 17 15 13] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 4
      },
      {
        "prompt": "[v w z l] &nbsp [v w z l o]<br>[4 6 8 10 12 14 16 18] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 5
      },
      {
        "prompt": "[c m t f] &nbsp [k m t f]<br>[20 20 0 0 21 21 0 0 22 22 0 0 23 23 0 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 6
      },
      {
        "prompt": "[x p j h] &nbsp [x p j h n]<br>[8 0 7 0 6 0 5 0] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 7
      },
      {
        "prompt": "[g w e v y] &nbsp [g y e v w]<br>[x x x x p p x x j j x x h h x x n n x x d d x x u u x x a a x x i i x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 8
      },
      {
        "prompt": "[x p j h] &nbsp [x p j n]<br>[p p x x h h x x a a x x d d x x] &nbsp [&nbsp ? &nbsp]",
        "prob_ind": 9
      }
    ]
  }
}