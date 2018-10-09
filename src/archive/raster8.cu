#include "raster8.hh"

float raster[3072] = {0.0000000, 0.0000000, 0.1963495, 0.0000000, 0.0000000, -0.1963495, 0.0000000, 0.0000000, 0.5890486, 0.4440014, 0.2563443, 0.2900492, -0.0000000, 0.5126887, 0.2900492, -0.4440014, 0.2563444, 0.2900492, -0.4440014, -0.2563444, 0.2900492, 0.0000000, -0.5126887, 0.2900492, 0.4440014, -0.2563444, 0.2900492, 0.0000000, 0.0000000, -0.5890486, 0.4440014, 0.2563443, -0.2900492, -0.0000000, 0.5126887, -0.2900492, -0.4440014, 0.2563444, -0.2900492, -0.4440014, -0.2563444, -0.2900492, 0.0000000, -0.5126887, -0.2900492, 0.4440014, -0.2563444, -0.2900492, 0.0000000, 0.0000000, 0.9817477, 0.4846523, 0.2798141, 0.8066255, -0.0000000, 0.5596282, 0.8066255, -0.4846523, 0.2798142, 0.8066255, -0.4846522, -0.2798142, 0.8066255, 0.0000000, -0.5596282, 0.8066255, 0.4846523, -0.2798142, 0.8066255, 0.8934412, 0.2393969, 0.3290601, 0.6540443, 0.6540443, 0.3290601, 0.2393969, 0.8934412, 0.3290601, -0.2393970, 0.8934412, 0.3290601, -0.6540443, 0.6540443, 0.3290601, -0.8934412, 0.2393969, 0.3290601, -0.8934412, -0.2393969, 0.3290601, -0.6540443, -0.6540444, 0.3290601, -0.2393968, -0.8934413, 0.3290601, 0.2393968, -0.8934412, 0.3290601, 0.6540443, -0.6540444, 0.3290601, 0.8934413, -0.2393967, 0.3290601, 0.0000000, 0.0000000, -0.9817477, 0.4846523, 0.2798141, -0.8066255, -0.0000000, 0.5596282, -0.8066255, -0.4846523, 0.2798142, -0.8066255, -0.4846522, -0.2798142, -0.8066255, 0.0000000, -0.5596282, -0.8066255, 0.4846523, -0.2798142, -0.8066255, 0.8934412, 0.2393969, -0.3290601, 0.6540443, 0.6540443, -0.3290601, 0.2393969, 0.8934412, -0.3290601, -0.2393970, 0.8934412, -0.3290601, -0.6540443, 0.6540443, -0.3290601, -0.8934412, 0.2393969, -0.3290601, -0.8934412, -0.2393969, -0.3290601, -0.6540443, -0.6540444, -0.3290601, -0.2393968, -0.8934413, -0.3290601, 0.2393968, -0.8934412, -0.3290601, 0.6540443, -0.6540444, -0.3290601, 0.8934413, -0.2393967, -0.3290601, 0.0000000, 0.0000000, 1.3744467, 0.4952744, 0.2859468, 1.2498165, -0.0000000, 0.5718936, 1.2498165, -0.4952744, 0.2859468, 1.2498165, -0.4952743, -0.2859469, 1.2498165, 0.0000000, -0.5718936, 1.2498165, 0.4952744, -0.2859469, 1.2498165, 0.9921317, 0.2658409, 0.9132946, 0.7262908, 0.7262908, 0.9132946, 0.2658409, 0.9921317, 0.9132946, -0.2658410, 0.9921317, 0.9132946, -0.7262908, 0.7262908, 0.9132946, -0.9921317, 0.2658410, 0.9132946, -0.9921317, -0.2658409, 0.9132946, -0.7262907, -0.7262909, 0.9132946, -0.2658408, -0.9921318, 0.9132946, 0.2658409, -0.9921317, 0.9132946, 0.7262907, -0.7262909, 0.9132946, 0.9921318, -0.2658407, 0.9132946, 1.3101258, 0.2310105, 0.3454101, 1.1521053, 0.6651683, 0.3454101, 0.8551240, 1.0190970, 0.3454101, 0.4550020, 1.2501075, 0.3454101, 0.0000001, 1.3303367, 0.3454101, -0.4550021, 1.2501075, 0.3454101, -0.8551240, 1.0190970, 0.3454101, -1.1521053, 0.6651684, 0.3454101, -1.3101258, 0.2310107, 0.3454101, -1.3101258, -0.2310106, 0.3454101, -1.1521053, -0.6651683, 0.3454101, -0.8551238, -1.0190972, 0.3454101, -0.4550018, -1.2501076, 0.3454101, 0.0000000, -1.3303367, 0.3454101, 0.4550019, -1.2501075, 0.3454101, 0.8551238, -1.0190971, 0.3454101, 1.1521052, -0.6651686, 0.3454101, 1.3101258, -0.2310109, 0.3454101, 0.0000000, 0.0000000, -1.3744467, 0.4952744, 0.2859468, -1.2498165, -0.0000000, 0.5718936, -1.2498165, -0.4952744, 0.2859468, -1.2498165, -0.4952743, -0.2859469, -1.2498165, 0.0000000, -0.5718936, -1.2498165, 0.4952744, -0.2859469, -1.2498165, 0.9921317, 0.2658409, -0.9132946, 0.7262908, 0.7262908, -0.9132946, 0.2658409, 0.9921317, -0.9132946, -0.2658410, 0.9921317, -0.9132946, -0.7262908, 0.7262908, -0.9132946, -0.9921317, 0.2658410, -0.9132946, -0.9921317, -0.2658409, -0.9132946, -0.7262907, -0.7262909, -0.9132946, -0.2658408, -0.9921318, -0.9132946, 0.2658409, -0.9921317, -0.9132946, 0.7262907, -0.7262909, -0.9132946, 0.9921318, -0.2658407, -0.9132946, 1.3101258, 0.2310105, -0.3454101, 1.1521053, 0.6651683, -0.3454101, 0.8551240, 1.0190970, -0.3454101, 0.4550020, 1.2501075, -0.3454101, 0.0000001, 1.3303367, -0.3454101, -0.4550021, 1.2501075, -0.3454101, -0.8551240, 1.0190970, -0.3454101, -1.1521053, 0.6651684, -0.3454101, -1.3101258, 0.2310107, -0.3454101, -1.3101258, -0.2310106, -0.3454101, -1.1521053, -0.6651683, -0.3454101, -0.8551238, -1.0190972, -0.3454101, -0.4550018, -1.2501076, -0.3454101, 0.0000000, -1.3303367, -0.3454101, 0.4550019, -1.2501075, -0.3454101, 0.8551238, -1.0190971, -0.3454101, 1.1521052, -0.6651686, -0.3454101, 1.3101258, -0.2310109, -0.3454101, 0.0000000, 0.0000000, 1.7671459, 0.4995990, 0.2884436, 1.6703310, -0.0000000, 0.5768872, 1.6703310, -0.4995990, 0.2884437, 1.6703310, -0.4995990, -0.2884437, 1.6703310, 0.0000000, -0.5768872, 1.6703310, 0.4995990, -0.2884437, 1.6703310, 1.0296917, 0.2759050, 1.4094027, 0.7537866, 0.7537866, 1.4094027, 0.2759051, 1.0296917, 1.4094027, -0.2759052, 1.0296917, 1.4094027, -0.7537866, 0.7537866, 1.4094027, -1.0296917, 0.2759051, 1.4094027, -1.0296917, -0.2759051, 1.4094027, -0.7537864, -0.7537867, 1.4094027, -0.2759050, -1.0296917, 1.4094027, 0.2759050, -1.0296917, 1.4094027, 0.7537864, -0.7537867, 1.4094027, 1.0296917, -0.2759048, 1.4094027, 1.4538680, 0.2563561, 0.9712641, 1.2785101, 0.7381482, 0.9712641, 0.9489450, 1.1309086, 0.9712641, 0.5049231, 1.3872647, 0.9712641, 0.0000001, 1.4762963, 0.9712641, -0.5049232, 1.3872647, 0.9712641, -0.9489450, 1.1309086, 0.9712641, -1.2785101, 0.7381482, 0.9712641, -1.4538680, 0.2563563, 0.9712641, -1.4538680, -0.2563562, 0.9712641, -1.2785101, -0.7381481, 0.9712641, -0.9489448, -1.1309087, 0.9712641, -0.5049230, -1.3872648, 0.9712641, 0.0000000, -1.4762963, 0.9712641, 0.5049230, -1.3872647, 0.9712641, 0.9489448, -1.1309086, 0.9712641, 1.2785100, -0.7381484, 0.9712641, 1.4538680, -0.2563566, 0.9712641, 1.7163386, 0.2259603, 0.3548640, 1.5993730, 0.6624820, 0.3548640, 1.3734127, 1.0538566, 0.3548640, 1.0538565, 1.3734127, 0.3548640, 0.6624820, 1.5993730, 0.3548640, 0.2259604, 1.7163385, 0.3548640, -0.2259603, 1.7163386, 0.3548640, -0.6624821, 1.5993730, 0.3548640, -1.0538566, 1.3734127, 0.3548640, -1.3734127, 1.0538566, 0.3548640, -1.5993730, 0.6624821, 0.3548640, -1.7163386, 0.2259601, 0.3548640, -1.7163385, -0.2259604, 0.3548640, -1.5993730, -0.6624820, 0.3548640, -1.3734127, -1.0538565, 0.3548640, -1.0538567, -1.3734126, 0.3548640, -0.6624822, -1.5993730, 0.3548640, -0.2259598, -1.7163386, 0.3548640, 0.2259598, -1.7163386, 0.3548640, 0.6624823, -1.5993729, 0.3548640, 1.0538561, -1.3734131, 0.3548640, 1.3734127, -1.0538565, 0.3548640, 1.5993730, -0.6624820, 0.3548640, 1.7163386, -0.2259603, 0.3548640, 0.0000000, 0.0000000, -1.7671459, 0.4995990, 0.2884436, -1.6703310, -0.0000000, 0.5768872, -1.6703310, -0.4995990, 0.2884437, -1.6703310, -0.4995990, -0.2884437, -1.6703310, 0.0000000, -0.5768872, -1.6703310, 0.4995990, -0.2884437, -1.6703310, 1.0296917, 0.2759050, -1.4094027, 0.7537866, 0.7537866, -1.4094027, 0.2759051, 1.0296917, -1.4094027, -0.2759052, 1.0296917, -1.4094027, -0.7537866, 0.7537866, -1.4094027, -1.0296917, 0.2759051, -1.4094027, -1.0296917, -0.2759051, -1.4094027, -0.7537864, -0.7537867, -1.4094027, -0.2759050, -1.0296917, -1.4094027, 0.2759050, -1.0296917, -1.4094027, 0.7537864, -0.7537867, -1.4094027, 1.0296917, -0.2759048, -1.4094027, 1.4538680, 0.2563561, -0.9712641, 1.2785101, 0.7381482, -0.9712641, 0.9489450, 1.1309086, -0.9712641, 0.5049231, 1.3872647, -0.9712641, 0.0000001, 1.4762963, -0.9712641, -0.5049232, 1.3872647, -0.9712641, -0.9489450, 1.1309086, -0.9712641, -1.2785101, 0.7381482, -0.9712641, -1.4538680, 0.2563563, -0.9712641, -1.4538680, -0.2563562, -0.9712641, -1.2785101, -0.7381481, -0.9712641, -0.9489448, -1.1309087, -0.9712641, -0.5049230, -1.3872648, -0.9712641, 0.0000000, -1.4762963, -0.9712641, 0.5049230, -1.3872647, -0.9712641, 0.9489448, -1.1309086, -0.9712641, 1.2785100, -0.7381484, -0.9712641, 1.4538680, -0.2563566, -0.9712641, 1.7163386, 0.2259603, -0.3548640, 1.5993730, 0.6624820, -0.3548640, 1.3734127, 1.0538566, -0.3548640, 1.0538565, 1.3734127, -0.3548640, 0.6624820, 1.5993730, -0.3548640, 0.2259604, 1.7163385, -0.3548640, -0.2259603, 1.7163386, -0.3548640, -0.6624821, 1.5993730, -0.3548640, -1.0538566, 1.3734127, -0.3548640, -1.3734127, 1.0538566, -0.3548640, -1.5993730, 0.6624821, -0.3548640, -1.7163386, 0.2259601, -0.3548640, -1.7163385, -0.2259604, -0.3548640, -1.5993730, -0.6624820, -0.3548640, -1.3734127, -1.0538565, -0.3548640, -1.0538567, -1.3734126, -0.3548640, -0.6624822, -1.5993730, -0.3548640, -0.2259598, -1.7163386, -0.3548640, 0.2259598, -1.7163386, -0.3548640, 0.6624823, -1.5993729, -0.3548640, 1.0538561, -1.3734131, -0.3548640, 1.3734127, -1.0538565, -0.3548640, 1.5993730, -0.6624820, -0.3548640, 1.7163386, -0.2259603, -0.3548640, 0.0000000, 0.0000000, 2.1598449, 0.5017790, 0.2897022, 2.0806777, -0.0000000, 0.5794045, 2.0806777, -0.5017790, 0.2897023, 2.0806777, -0.5017789, -0.2897023, 2.0806777, 0.0000000, -0.5794045, 2.0806777, 0.5017790, -0.2897023, 2.0806777, 1.0481859, 0.2808606, 1.8674458, 0.7673253, 0.7673253, 1.8674458, 0.2808606, 1.0481859, 1.8674458, -0.2808607, 1.0481859, 1.8674458, -0.7673253, 0.7673253, 1.8674458, -1.0481859, 0.2808607, 1.8674458, -1.0481859, -0.2808606, 1.8674458, -0.7673252, -0.7673255, 1.8674458, -0.2808605, -1.0481859, 1.8674458, 0.2808605, -1.0481859, 1.8674458, 0.7673252, -0.7673255, 1.8674458, 1.0481859, -0.2808603, 1.8674458, 1.5210335, 0.2681993, 1.5097867, 1.3375745, 0.7722490, 1.5097867, 0.9927842, 1.1831541, 1.5097867, 0.5282494, 1.4513533, 1.5097867, 0.0000001, 1.5444980, 1.5097867, -0.5282496, 1.4513533, 1.5097867, -0.9927842, 1.1831541, 1.5097867, -1.3375745, 0.7722491, 1.5097867, -1.5210335, 0.2681994, 1.5097867, -1.5210335, -0.2681993, 1.5097867, -1.3375745, -0.7722489, 1.5097867, -0.9927840, -1.1831542, 1.5097867, -0.5282493, -1.4513534, 1.5097867, 0.0000000, -1.5444980, 1.5097867, 0.5282493, -1.4513533, 1.5097867, 0.9927840, -1.1831542, 1.5097867, 1.3375744, -0.7722493, 1.5097867, 1.5210335, -0.2681997, 1.5097867, 1.8937950, 0.2493228, 1.0081213, 1.7647359, 0.7309776, 1.0081213, 1.5154130, 1.1628174, 1.0081213, 1.1628172, 1.5154132, 1.0081213, 0.7309776, 1.7647359, 1.0081213, 0.2493229, 1.8937949, 1.0081213, -0.2493229, 1.8937950, 1.0081213, -0.7309777, 1.7647359, 1.0081213, -1.1628174, 1.5154130, 1.0081213, -1.5154130, 1.1628174, 1.0081213, -1.7647359, 0.7309777, 1.0081213, -1.8937950, 0.2493226, 1.0081213, -1.8937949, -0.2493230, 1.0081213, -1.7647359, -0.7309776, 1.0081213, -1.5154132, -1.1628172, 1.0081213, -1.1628175, -1.5154130, 1.0081213, -0.7309778, -1.7647359, 1.0081213, -0.2493223, -1.8937950, 1.0081213, 0.2493224, -1.8937950, 1.0081213, 0.7309779, -1.7647358, 1.0081213, 1.1628169, -1.5154135, 1.0081213, 1.5154132, -1.1628172, 1.0081213, 1.7647361, -0.7309776, 1.0081213, 1.8937950, -0.2493229, 1.0081213, 2.1177797, 0.2225876, 0.3611009, 2.0252228, 0.6580347, 0.3611009, 1.8441535, 1.0647225, 0.3611009, 1.5824860, 1.4248769, 0.3611009, 1.2516564, 1.7227572, 0.3611009, 0.8661233, 1.9453449, 0.3611009, 0.4427364, 2.0829117, 0.3611009, -0.0000001, 2.1294451, 0.3611009, -0.4427367, 2.0829115, 0.3611009, -0.8661234, 1.9453448, 0.3611009, -1.2516563, 1.7227572, 0.3611009, -1.5824863, 1.4248766, 0.3611009, -1.8441535, 1.0647227, 0.3611009, -2.0252228, 0.6580343, 0.3611009, -2.1177797, 0.2225877, 0.3611009, -2.1177797, -0.2225881, 0.3611009, -2.0252228, -0.6580347, 0.3611009, -1.8441533, -1.0647229, 0.3611009, -1.5824860, -1.4248769, 0.3611009, -1.2516561, -1.7227576, 0.3611009, -0.8661233, -1.9453448, 0.3611009, -0.4427360, -2.0829117, 0.3611009, 0.0000000, -2.1294451, 0.3611009, 0.4427361, -2.0829117, 0.3611009, 0.8661234, -1.9453448, 0.3611009, 1.2516569, -1.7227569, 0.3611009, 1.5824862, -1.4248768, 0.3611009, 1.8441534, -1.0647229, 0.3611009, 2.0252228, -0.6580346, 0.3611009, 2.1177797, -0.2225870, 0.3611009, 0.0000000, 0.0000000, -2.1598449, 0.5017790, 0.2897022, -2.0806777, -0.0000000, 0.5794045, -2.0806777, -0.5017790, 0.2897023, -2.0806777, -0.5017789, -0.2897023, -2.0806777, 0.0000000, -0.5794045, -2.0806777, 0.5017790, -0.2897023, -2.0806777, 1.0481859, 0.2808606, -1.8674458, 0.7673253, 0.7673253, -1.8674458, 0.2808606, 1.0481859, -1.8674458, -0.2808607, 1.0481859, -1.8674458, -0.7673253, 0.7673253, -1.8674458, -1.0481859, 0.2808607, -1.8674458, -1.0481859, -0.2808606, -1.8674458, -0.7673252, -0.7673255, -1.8674458, -0.2808605, -1.0481859, -1.8674458, 0.2808605, -1.0481859, -1.8674458, 0.7673252, -0.7673255, -1.8674458, 1.0481859, -0.2808603, -1.8674458, 1.5210335, 0.2681993, -1.5097867, 1.3375745, 0.7722490, -1.5097867, 0.9927842, 1.1831541, -1.5097867, 0.5282494, 1.4513533, -1.5097867, 0.0000001, 1.5444980, -1.5097867, -0.5282496, 1.4513533, -1.5097867, -0.9927842, 1.1831541, -1.5097867, -1.3375745, 0.7722491, -1.5097867, -1.5210335, 0.2681994, -1.5097867, -1.5210335, -0.2681993, -1.5097867, -1.3375745, -0.7722489, -1.5097867, -0.9927840, -1.1831542, -1.5097867, -0.5282493, -1.4513534, -1.5097867, 0.0000000, -1.5444980, -1.5097867, 0.5282493, -1.4513533, -1.5097867, 0.9927840, -1.1831542, -1.5097867, 1.3375744, -0.7722493, -1.5097867, 1.5210335, -0.2681997, -1.5097867, 1.8937950, 0.2493228, -1.0081213, 1.7647359, 0.7309776, -1.0081213, 1.5154130, 1.1628174, -1.0081213, 1.1628172, 1.5154132, -1.0081213, 0.7309776, 1.7647359, -1.0081213, 0.2493229, 1.8937949, -1.0081213, -0.2493229, 1.8937950, -1.0081213, -0.7309777, 1.7647359, -1.0081213, -1.1628174, 1.5154130, -1.0081213, -1.5154130, 1.1628174, -1.0081213, -1.7647359, 0.7309777, -1.0081213, -1.8937950, 0.2493226, -1.0081213, -1.8937949, -0.2493230, -1.0081213, -1.7647359, -0.7309776, -1.0081213, -1.5154132, -1.1628172, -1.0081213, -1.1628175, -1.5154130, -1.0081213, -0.7309778, -1.7647359, -1.0081213, -0.2493223, -1.8937950, -1.0081213, 0.2493224, -1.8937950, -1.0081213, 0.7309779, -1.7647358, -1.0081213, 1.1628169, -1.5154135, -1.0081213, 1.5154132, -1.1628172, -1.0081213, 1.7647361, -0.7309776, -1.0081213, 1.8937950, -0.2493229, -1.0081213, 2.1177797, 0.2225876, -0.3611009, 2.0252228, 0.6580347, -0.3611009, 1.8441535, 1.0647225, -0.3611009, 1.5824860, 1.4248769, -0.3611009, 1.2516564, 1.7227572, -0.3611009, 0.8661233, 1.9453449, -0.3611009, 0.4427364, 2.0829117, -0.3611009, -0.0000001, 2.1294451, -0.3611009, -0.4427367, 2.0829115, -0.3611009, -0.8661234, 1.9453448, -0.3611009, -1.2516563, 1.7227572, -0.3611009, -1.5824863, 1.4248766, -0.3611009, -1.8441535, 1.0647227, -0.3611009, -2.0252228, 0.6580343, -0.3611009, -2.1177797, 0.2225877, -0.3611009, -2.1177797, -0.2225881, -0.3611009, -2.0252228, -0.6580347, -0.3611009, -1.8441533, -1.0647229, -0.3611009, -1.5824860, -1.4248769, -0.3611009, -1.2516561, -1.7227576, -0.3611009, -0.8661233, -1.9453448, -0.3611009, -0.4427360, -2.0829117, -0.3611009, 0.0000000, -2.1294451, -0.3611009, 0.4427361, -2.0829117, -0.3611009, 0.8661234, -1.9453448, -0.3611009, 1.2516569, -1.7227569, -0.3611009, 1.5824862, -1.4248768, -0.3611009, 1.8441534, -1.0647229, -0.3611009, 2.0252228, -0.6580346, -0.3611009, 2.1177797, -0.2225870, -0.3611009, 0.0000000, 0.0000000, 2.5525441, 0.5030303, 0.2904247, 2.4855776, -0.0000000, 0.5808493, 2.4855776, -0.5030303, 0.2904247, 2.4855776, -0.5030302, -0.2904248, 2.4855776, 0.0000000, -0.5808493, 2.4855776, 0.5030302, -0.2904248, 2.4855776, 1.0586808, 0.2836727, 2.3052561, 0.7750081, 0.7750081, 2.3052561, 0.2836727, 1.0586808, 2.3052561, -0.2836728, 1.0586808, 2.3052561, -0.7750081, 0.7750081, 2.3052561, -1.0586808, 0.2836727, 2.3052561, -1.0586808, -0.2836727, 2.3052561, -0.7750080, -0.7750083, 2.3052561, -0.2836726, -1.0586809, 2.3052561, 0.2836726, -1.0586808, 2.3052561, 0.7750080, -0.7750083, 2.3052561, 1.0586809, -0.2836724, 2.3052561, 1.5583125, 0.2747726, 2.0029087, 1.3703570, 0.7911760, 2.0029087, 1.0171163, 1.2121520, 2.0029087, 0.5411963, 1.4869245, 2.0029087, 0.0000001, 1.5823520, 2.0029087, -0.5411964, 1.4869245, 2.0029087, -1.0171163, 1.2121520, 2.0029087, -1.3703570, 0.7911761, 2.0029087, -1.5583125, 0.2747727, 2.0029087, -1.5583125, -0.2747726, 2.0029087, -1.3703570, -0.7911760, 2.0029087, -1.0171162, -1.2121521, 2.0029087, -0.5411962, -1.4869246, 2.0029087, 0.0000000, -1.5823520, 2.0029087, 0.5411962, -1.4869245, 2.0029087, 1.0171162, -1.2121521, 2.0029087, 1.3703569, -0.7911763, 2.0029087, 1.5583125, -0.2747730, 2.0029087, 1.9882648, 0.2617600, 1.5791664, 1.8527677, 0.7674416, 1.5791664, 1.5910077, 1.2208232, 1.5791664, 1.2208230, 1.5910078, 1.5791664, 0.7674415, 1.8527677, 1.5791664, 0.2617601, 1.9882647, 1.5791664, -0.2617601, 1.9882648, 1.5791664, -0.7674417, 1.8527677, 1.5791664, -1.2208232, 1.5910077, 1.5791664, -1.5910077, 1.2208232, 1.5791664, -1.8527677, 0.7674416, 1.5791664, -1.9882648, 0.2617598, 1.5791664, -1.9882647, -0.2617601, 1.5791664, -1.8527677, -0.7674415, 1.5791664, -1.5910078, -1.2208230, 1.5791664, -1.2208233, -1.5910076, 1.5791664, -0.7674418, -1.8527677, 1.5791664, -0.2617595, -1.9882648, 1.5791664, 0.2617595, -1.9882648, 1.5791664, 0.7674419, -1.8527676, 1.5791664, 1.2208226, -1.5910082, 1.5791664, 1.5910078, -1.2208230, 1.5791664, 1.8527678, -0.7674415, 1.5791664, 1.9882648, -0.2617601, 1.5791664, 2.3210850, 0.2439559, 1.0336982, 2.2196424, 0.7212056, 1.0336982, 2.0211909, 1.1669351, 1.0336982, 1.7344035, 1.5616640, 1.0336982, 1.3718145, 1.8881407, 1.0336982, 0.9492704, 2.1320965, 1.0336982, 0.4852388, 2.2828696, 1.0336982, -0.0000001, 2.3338702, 1.0336982, -0.4852390, 2.2828693, 1.0336982, -0.9492707, 2.1320965, 1.0336982, -1.3718143, 1.8881407, 1.0336982, -1.7344037, 1.5616636, 1.0336982, -2.0211909, 1.1669352, 1.0336982, -2.2196426, 0.7212051, 1.0336982, -2.3210850, 0.2439559, 1.0336982, -2.3210850, -0.2439564, 1.0336982, -2.2196424, -0.7212055, 1.0336982, -2.0211906, -1.1669356, 1.0336982, -1.7344035, -1.5616640, 1.0336982, -1.3718140, -1.8881409, 1.0336982, -0.9492705, -2.1320965, 1.0336982, -0.4852383, -2.2828696, 1.0336982, 0.0000000, -2.3338702, 1.0336982, 0.4852384, -2.2828696, 1.0336982, 0.9492706, -2.1320965, 1.0336982, 1.3718150, -1.8881402, 1.0336982, 1.7344036, -1.5616639, 1.0336982, 2.0211906, -1.1669356, 1.0336982, 2.2196424, -0.7212054, 1.0336982, 2.3210850, -0.2439552, 1.0336982, 2.5166209, 0.2201758, 0.3655457, 2.4401546, 0.6538374, 0.3655457, 2.2895453, 1.0676326, 0.3655457, 2.0693698, 1.4489883, 0.3655457, 1.7863172, 1.7863171, 0.3655457, 1.4489881, 2.0693698, 0.3655457, 1.0676326, 2.2895455, 0.3655457, 0.6538375, 2.4401546, 0.3655457, 0.2201760, 2.5166209, 0.3655457, -0.2201758, 2.5166209, 0.3655457, -0.6538374, 2.4401546, 0.3655457, -1.0676328, 2.2895453, 0.3655457, -1.4489883, 2.0693696, 0.3655457, -1.7863171, 1.7863171, 0.3655457, -2.0693698, 1.4489883, 0.3655457, -2.2895453, 1.0676328, 0.3655457, -2.4401546, 0.6538377, 0.3655457, -2.5166206, 0.2201761, 0.3655457, -2.5166209, -0.2201760, 0.3655457, -2.4401546, -0.6538375, 0.3655457, -2.2895455, -1.0676326, 0.3655457, -2.0693698, -1.4489881, 0.3655457, -1.7863168, -1.7863173, 0.3655457, -1.4489880, -2.0693698, 0.3655457, -1.0676324, -2.2895455, 0.3655457, -0.6538373, -2.4401548, 0.3655457, -0.2201757, -2.5166209, 0.3655457, 0.2201758, -2.5166209, 0.3655457, 0.6538374, -2.4401546, 0.3655457, 1.0676324, -2.2895455, 0.3655457, 1.4489880, -2.0693698, 0.3655457, 1.7863168, -1.7863173, 0.3655457, 2.0693693, -1.4489886, 0.3655457, 2.2895453, -1.0676330, 0.3655457, 2.4401548, -0.6538369, 0.3655457, 2.5166209, -0.2201753, 0.3655457, 0.0000000, 0.0000000, -2.5525441, 0.5030303, 0.2904247, -2.4855776, -0.0000000, 0.5808493, -2.4855776, -0.5030303, 0.2904247, -2.4855776, -0.5030302, -0.2904248, -2.4855776, 0.0000000, -0.5808493, -2.4855776, 0.5030302, -0.2904248, -2.4855776, 1.0586808, 0.2836727, -2.3052561, 0.7750081, 0.7750081, -2.3052561, 0.2836727, 1.0586808, -2.3052561, -0.2836728, 1.0586808, -2.3052561, -0.7750081, 0.7750081, -2.3052561, -1.0586808, 0.2836727, -2.3052561, -1.0586808, -0.2836727, -2.3052561, -0.7750080, -0.7750083, -2.3052561, -0.2836726, -1.0586809, -2.3052561, 0.2836726, -1.0586808, -2.3052561, 0.7750080, -0.7750083, -2.3052561, 1.0586809, -0.2836724, -2.3052561, 1.5583125, 0.2747726, -2.0029087, 1.3703570, 0.7911760, -2.0029087, 1.0171163, 1.2121520, -2.0029087, 0.5411963, 1.4869245, -2.0029087, 0.0000001, 1.5823520, -2.0029087, -0.5411964, 1.4869245, -2.0029087, -1.0171163, 1.2121520, -2.0029087, -1.3703570, 0.7911761, -2.0029087, -1.5583125, 0.2747727, -2.0029087, -1.5583125, -0.2747726, -2.0029087, -1.3703570, -0.7911760, -2.0029087, -1.0171162, -1.2121521, -2.0029087, -0.5411962, -1.4869246, -2.0029087, 0.0000000, -1.5823520, -2.0029087, 0.5411962, -1.4869245, -2.0029087, 1.0171162, -1.2121521, -2.0029087, 1.3703569, -0.7911763, -2.0029087, 1.5583125, -0.2747730, -2.0029087, 1.9882648, 0.2617600, -1.5791664, 1.8527677, 0.7674416, -1.5791664, 1.5910077, 1.2208232, -1.5791664, 1.2208230, 1.5910078, -1.5791664, 0.7674415, 1.8527677, -1.5791664, 0.2617601, 1.9882647, -1.5791664, -0.2617601, 1.9882648, -1.5791664, -0.7674417, 1.8527677, -1.5791664, -1.2208232, 1.5910077, -1.5791664, -1.5910077, 1.2208232, -1.5791664, -1.8527677, 0.7674416, -1.5791664, -1.9882648, 0.2617598, -1.5791664, -1.9882647, -0.2617601, -1.5791664, -1.8527677, -0.7674415, -1.5791664, -1.5910078, -1.2208230, -1.5791664, -1.2208233, -1.5910076, -1.5791664, -0.7674418, -1.8527677, -1.5791664, -0.2617595, -1.9882648, -1.5791664, 0.2617595, -1.9882648, -1.5791664, 0.7674419, -1.8527676, -1.5791664, 1.2208226, -1.5910082, -1.5791664, 1.5910078, -1.2208230, -1.5791664, 1.8527678, -0.7674415, -1.5791664, 1.9882648, -0.2617601, -1.5791664, 2.3210850, 0.2439559, -1.0336982, 2.2196424, 0.7212056, -1.0336982, 2.0211909, 1.1669351, -1.0336982, 1.7344035, 1.5616640, -1.0336982, 1.3718145, 1.8881407, -1.0336982, 0.9492704, 2.1320965, -1.0336982, 0.4852388, 2.2828696, -1.0336982, -0.0000001, 2.3338702, -1.0336982, -0.4852390, 2.2828693, -1.0336982, -0.9492707, 2.1320965, -1.0336982, -1.3718143, 1.8881407, -1.0336982, -1.7344037, 1.5616636, -1.0336982, -2.0211909, 1.1669352, -1.0336982, -2.2196426, 0.7212051, -1.0336982, -2.3210850, 0.2439559, -1.0336982, -2.3210850, -0.2439564, -1.0336982, -2.2196424, -0.7212055, -1.0336982, -2.0211906, -1.1669356, -1.0336982, -1.7344035, -1.5616640, -1.0336982, -1.3718140, -1.8881409, -1.0336982, -0.9492705, -2.1320965, -1.0336982, -0.4852383, -2.2828696, -1.0336982, 0.0000000, -2.3338702, -1.0336982, 0.4852384, -2.2828696, -1.0336982, 0.9492706, -2.1320965, -1.0336982, 1.3718150, -1.8881402, -1.0336982, 1.7344036, -1.5616639, -1.0336982, 2.0211906, -1.1669356, -1.0336982, 2.2196424, -0.7212054, -1.0336982, 2.3210850, -0.2439552, -1.0336982, 2.5166209, 0.2201758, -0.3655457, 2.4401546, 0.6538374, -0.3655457, 2.2895453, 1.0676326, -0.3655457, 2.0693698, 1.4489883, -0.3655457, 1.7863172, 1.7863171, -0.3655457, 1.4489881, 2.0693698, -0.3655457, 1.0676326, 2.2895455, -0.3655457, 0.6538375, 2.4401546, -0.3655457, 0.2201760, 2.5166209, -0.3655457, -0.2201758, 2.5166209, -0.3655457, -0.6538374, 2.4401546, -0.3655457, -1.0676328, 2.2895453, -0.3655457, -1.4489883, 2.0693696, -0.3655457, -1.7863171, 1.7863171, -0.3655457, -2.0693698, 1.4489883, -0.3655457, -2.2895453, 1.0676328, -0.3655457, -2.4401546, 0.6538377, -0.3655457, -2.5166206, 0.2201761, -0.3655457, -2.5166209, -0.2201760, -0.3655457, -2.4401546, -0.6538375, -0.3655457, -2.2895455, -1.0676326, -0.3655457, -2.0693698, -1.4489881, -0.3655457, -1.7863168, -1.7863173, -0.3655457, -1.4489880, -2.0693698, -0.3655457, -1.0676324, -2.2895455, -0.3655457, -0.6538373, -2.4401548, -0.3655457, -0.2201757, -2.5166209, -0.3655457, 0.2201758, -2.5166209, -0.3655457, 0.6538374, -2.4401546, -0.3655457, 1.0676324, -2.2895455, -0.3655457, 1.4489880, -2.0693698, -0.3655457, 1.7863168, -1.7863173, -0.3655457, 2.0693693, -1.4489886, -0.3655457, 2.2895453, -1.0676330, -0.3655457, 2.4401548, -0.6538369, -0.3655457, 2.5166209, -0.2201753, -0.3655457, 0.0000000, 0.0000000, 2.9452431, 0.5038143, 0.2908773, 2.8872163, -0.0000000, 0.5817547, 2.8872163, -0.5038143, 0.2908774, 2.8872163, -0.5038143, -0.2908774, 2.8872163, 0.0000000, -0.5817547, 2.8872163, 0.5038143, -0.2908774, 2.8872163, 1.0652151, 0.2854235, 2.7309902, 0.7797915, 0.7797915, 2.7309902, 0.2854235, 1.0652151, 2.7309902, -0.2854236, 1.0652151, 2.7309902, -0.7797915, 0.7797915, 2.7309902, -1.0652151, 0.2854236, 2.7309902, -1.0652151, -0.2854235, 2.7309902, -0.7797914, -0.7797917, 2.7309902, -0.2854235, -1.0652151, 2.7309902, 0.2854235, -1.0652151, 2.7309902, 0.7797914, -0.7797917, 2.7309902, 1.0652151, -0.2854233, 2.7309902, 1.5812492, 0.2788169, 2.4690826, 1.3905272, 0.8028213, 2.4690826, 1.0320872, 1.2299936, 2.4690826, 0.5491621, 1.5088104, 2.4690826, 0.0000001, 1.6056426, 2.4690826, -0.5491623, 1.5088104, 2.4690826, -1.0320872, 1.2299936, 2.4690826, -1.3905272, 0.8028214, 2.4690826, -1.5812492, 0.2788171, 2.4690826, -1.5812492, -0.2788170, 2.4690826, -1.3905272, -0.8028212, 2.4690826, -1.0320870, -1.2299937, 2.4690826, -0.5491620, -1.5088105, 2.4690826, 0.0000000, -1.6056426, 2.4690826, 0.5491620, -1.5088104, 2.4690826, 1.0320870, -1.2299937, 2.4690826, 1.3905271, -0.8028216, 2.4690826, 1.5812492, -0.2788173, 2.4690826, 2.0452428, 0.2692613, 2.1021268, 1.9058627, 0.7894343, 2.1021268, 1.6366014, 1.2558085, 2.1021268, 1.2558084, 1.6366016, 2.1021268, 0.7894342, 1.9058627, 2.1021268, 0.2692614, 2.0452425, 2.1021268, -0.2692614, 2.0452428, 2.1021268, -0.7894344, 1.9058627, 2.1021268, -1.2558085, 1.6366014, 2.1021268, -1.6366014, 1.2558085, 2.1021268, -1.9058627, 0.7894343, 2.1021268, -2.0452428, 0.2692611, 2.1021268, -2.0452425, -0.2692614, 2.1021268, -1.9058627, -0.7894342, 2.1021268, -1.6366016, -1.2558084, 2.1021268, -1.2558086, -1.6366013, 2.1021268, -0.7894345, -1.9058627, 2.1021268, -0.2692607, -2.0452428, 2.1021268, 0.2692608, -2.0452428, 2.1021268, 0.7894346, -1.9058626, 2.1021268, 1.2558079, -1.6366019, 2.1021268, 1.6366016, -1.2558084, 2.1021268, 1.9058628, -0.7894342, 2.1021268, 2.0452428, -0.2692614, 2.1021268, 2.4396081, 0.2564131, 1.6300372, 2.3329856, 0.7580330, 1.6300372, 2.1244001, 1.2265230, 1.6300372, 1.8229685, 1.6414082, 1.6300372, 1.4418643, 1.9845560, 1.6300372, 0.9977437, 2.2409692, 1.6300372, 0.5100169, 2.3994412, 1.6300372, -0.0000001, 2.4530461, 1.6300372, -0.5100171, 2.3994410, 1.6300372, -0.9977438, 2.2409689, 1.6300372, -1.4418641, 1.9845560, 1.6300372, -1.8229687, 1.6414078, 1.6300372, -2.1244001, 1.2265232, 1.6300372, -2.3329856, 0.7580324, 1.6300372, -2.4396081, 0.2564132, 1.6300372, -2.4396079, -0.2564136, 1.6300372, -2.3329856, -0.7580329, 1.6300372, -2.1243999, -1.2265235, 1.6300372, -1.8229685, -1.6414082, 1.6300372, -1.4418639, -1.9845563, 1.6300372, -0.9977437, -2.2409689, 1.6300372, -0.5100164, -2.3994412, 1.6300372, 0.0000000, -2.4530461, 1.6300372, 0.5100164, -2.3994412, 1.6300372, 0.9977438, -2.2409689, 1.6300372, 1.4418648, -1.9845556, 1.6300372, 1.8229686, -1.6414081, 1.6300372, 2.1244001, -1.2265235, 1.6300372, 2.3329856, -0.7580328, 1.6300372, 2.4396081, -0.2564124, 1.6300372, 2.7402909, 0.2397444, 1.0525147, 2.6570284, 0.7119486, 1.0525147, 2.4930334, 1.1625206, 1.0525147, 2.2532892, 1.5777701, 1.0525147, 1.9450799, 1.9450797, 1.0525147, 1.5777700, 2.2532892, 1.0525147, 1.1625205, 2.4930336, 1.0525147, 0.7119487, 2.6570284, 1.0525147, 0.2397445, 2.7402909, 1.0525147, -0.2397444, 2.7402909, 1.0525147, -0.7119486, 2.6570284, 1.0525147, -1.1625208, 2.4930334, 1.0525147, -1.5777701, 2.2532890, 1.0525147, -1.9450797, 1.9450797, 1.0525147, -2.2532892, 1.5777701, 1.0525147, -2.4930334, 1.1625208, 1.0525147, -2.6570284, 0.7119489, 1.0525147, -2.7402906, 0.2397447, 1.0525147, -2.7402909, -0.2397446, 1.0525147, -2.6570284, -0.7119487, 1.0525147, -2.4930336, -1.1625206, 1.0525147, -2.2532892, -1.5777700, 1.0525147, -1.9450794, -1.9450800, 1.0525147, -1.5777698, -2.2532895, 1.0525147, -1.1625204, -2.4930336, 1.0525147, -0.7119485, -2.6570284, 1.0525147, -0.2397443, -2.7402909, 1.0525147, 0.2397444, -2.7402909, 1.0525147, 0.7119485, -2.6570284, 1.0525147, 1.1625204, -2.4930336, 1.0525147, 1.5777698, -2.2532895, 1.0525147, 1.9450794, -1.9450800, 1.0525147, 2.2532890, -1.5777705, 1.0525147, 2.4930334, -1.1625211, 1.0525147, 2.6570284, -0.7119480, 1.0525147, 2.7402909, -0.2397438, 1.0525147, 2.9138803, 0.2183652, 0.3688816, 2.8487892, 0.6502175, 0.3688816, 2.7200608, 1.0675452, 0.3688816, 2.5305705, 1.4610256, 0.3688816, 2.2845514, 1.8218693, 0.3688816, 1.9874996, 2.1420152, 0.3688816, 1.6460500, 2.4143121, 0.3688816, 1.2678305, 2.6326771, 0.3688816, 0.8612897, 2.7922325, 0.3688816, 0.4355091, 2.8894143, 0.3688816, 0.0000002, 2.9220512, 0.3688816, -0.4355093, 2.8894143, 0.3688816, -0.8612896, 2.7922328, 0.3688816, -1.2678308, 2.6326771, 0.3688816, -1.6460500, 2.4143119, 0.3688816, -1.9875000, 2.1420147, 0.3688816, -2.2845516, 1.8218689, 0.3688816, -2.5305705, 1.4610257, 0.3688816, -2.7200606, 1.0675455, 0.3688816, -2.8487892, 0.6502175, 0.3688816, -2.9138806, 0.2183648, 0.3688816, -2.9138803, -0.2183653, 0.3688816, -2.8487892, -0.6502174, 0.3688816, -2.7200608, -1.0675448, 0.3688816, -2.5305705, -1.4610255, 0.3688816, -2.2845514, -1.8218695, 0.3688816, -1.9874996, -2.1420152, 0.3688816, -1.6460491, -2.4143126, 0.3688816, -1.2678310, -2.6326768, 0.3688816, -0.8612891, -2.7922328, 0.3688816, -0.4355102, -2.8894141, 0.3688816, 0.0000000, -2.9220512, 0.3688816, 0.4355102, -2.8894141, 0.3688816, 0.8612893, -2.7922328, 0.3688816, 1.2678310, -2.6326768, 0.3688816, 1.6460491, -2.4143126, 0.3688816, 1.9874996, -2.1420152, 0.3688816, 2.2845523, -1.8218682, 0.3688816, 2.5305703, -1.4610261, 0.3688816, 2.7200608, -1.0675447, 0.3688816, 2.8487890, -0.6502187, 0.3688816, 2.9138803, -0.2183652, 0.3688816, 0.0000000, 0.0000000, -2.9452431, 0.5038143, 0.2908773, -2.8872163, -0.0000000, 0.5817547, -2.8872163, -0.5038143, 0.2908774, -2.8872163, -0.5038143, -0.2908774, -2.8872163, 0.0000000, -0.5817547, -2.8872163, 0.5038143, -0.2908774, -2.8872163, 1.0652151, 0.2854235, -2.7309902, 0.7797915, 0.7797915, -2.7309902, 0.2854235, 1.0652151, -2.7309902, -0.2854236, 1.0652151, -2.7309902, -0.7797915, 0.7797915, -2.7309902, -1.0652151, 0.2854236, -2.7309902, -1.0652151, -0.2854235, -2.7309902, -0.7797914, -0.7797917, -2.7309902, -0.2854235, -1.0652151, -2.7309902, 0.2854235, -1.0652151, -2.7309902, 0.7797914, -0.7797917, -2.7309902, 1.0652151, -0.2854233, -2.7309902, 1.5812492, 0.2788169, -2.4690826, 1.3905272, 0.8028213, -2.4690826, 1.0320872, 1.2299936, -2.4690826, 0.5491621, 1.5088104, -2.4690826, 0.0000001, 1.6056426, -2.4690826, -0.5491623, 1.5088104, -2.4690826, -1.0320872, 1.2299936, -2.4690826, -1.3905272, 0.8028214, -2.4690826, -1.5812492, 0.2788171, -2.4690826, -1.5812492, -0.2788170, -2.4690826, -1.3905272, -0.8028212, -2.4690826, -1.0320870, -1.2299937, -2.4690826, -0.5491620, -1.5088105, -2.4690826, 0.0000000, -1.6056426, -2.4690826, 0.5491620, -1.5088104, -2.4690826, 1.0320870, -1.2299937, -2.4690826, 1.3905271, -0.8028216, -2.4690826, 1.5812492, -0.2788173, -2.4690826, 2.0452428, 0.2692613, -2.1021268, 1.9058627, 0.7894343, -2.1021268, 1.6366014, 1.2558085, -2.1021268, 1.2558084, 1.6366016, -2.1021268, 0.7894342, 1.9058627, -2.1021268, 0.2692614, 2.0452425, -2.1021268, -0.2692614, 2.0452428, -2.1021268, -0.7894344, 1.9058627, -2.1021268, -1.2558085, 1.6366014, -2.1021268, -1.6366014, 1.2558085, -2.1021268, -1.9058627, 0.7894343, -2.1021268, -2.0452428, 0.2692611, -2.1021268, -2.0452425, -0.2692614, -2.1021268, -1.9058627, -0.7894342, -2.1021268, -1.6366016, -1.2558084, -2.1021268, -1.2558086, -1.6366013, -2.1021268, -0.7894345, -1.9058627, -2.1021268, -0.2692607, -2.0452428, -2.1021268, 0.2692608, -2.0452428, -2.1021268, 0.7894346, -1.9058626, -2.1021268, 1.2558079, -1.6366019, -2.1021268, 1.6366016, -1.2558084, -2.1021268, 1.9058628, -0.7894342, -2.1021268, 2.0452428, -0.2692614, -2.1021268, 2.4396081, 0.2564131, -1.6300372, 2.3329856, 0.7580330, -1.6300372, 2.1244001, 1.2265230, -1.6300372, 1.8229685, 1.6414082, -1.6300372, 1.4418643, 1.9845560, -1.6300372, 0.9977437, 2.2409692, -1.6300372, 0.5100169, 2.3994412, -1.6300372, -0.0000001, 2.4530461, -1.6300372, -0.5100171, 2.3994410, -1.6300372, -0.9977438, 2.2409689, -1.6300372, -1.4418641, 1.9845560, -1.6300372, -1.8229687, 1.6414078, -1.6300372, -2.1244001, 1.2265232, -1.6300372, -2.3329856, 0.7580324, -1.6300372, -2.4396081, 0.2564132, -1.6300372, -2.4396079, -0.2564136, -1.6300372, -2.3329856, -0.7580329, -1.6300372, -2.1243999, -1.2265235, -1.6300372, -1.8229685, -1.6414082, -1.6300372, -1.4418639, -1.9845563, -1.6300372, -0.9977437, -2.2409689, -1.6300372, -0.5100164, -2.3994412, -1.6300372, 0.0000000, -2.4530461, -1.6300372, 0.5100164, -2.3994412, -1.6300372, 0.9977438, -2.2409689, -1.6300372, 1.4418648, -1.9845556, -1.6300372, 1.8229686, -1.6414081, -1.6300372, 2.1244001, -1.2265235, -1.6300372, 2.3329856, -0.7580328, -1.6300372, 2.4396081, -0.2564124, -1.6300372, 2.7402909, 0.2397444, -1.0525147, 2.6570284, 0.7119486, -1.0525147, 2.4930334, 1.1625206, -1.0525147, 2.2532892, 1.5777701, -1.0525147, 1.9450799, 1.9450797, -1.0525147, 1.5777700, 2.2532892, -1.0525147, 1.1625205, 2.4930336, -1.0525147, 0.7119487, 2.6570284, -1.0525147, 0.2397445, 2.7402909, -1.0525147, -0.2397444, 2.7402909, -1.0525147, -0.7119486, 2.6570284, -1.0525147, -1.1625208, 2.4930334, -1.0525147, -1.5777701, 2.2532890, -1.0525147, -1.9450797, 1.9450797, -1.0525147, -2.2532892, 1.5777701, -1.0525147, -2.4930334, 1.1625208, -1.0525147, -2.6570284, 0.7119489, -1.0525147, -2.7402906, 0.2397447, -1.0525147, -2.7402909, -0.2397446, -1.0525147, -2.6570284, -0.7119487, -1.0525147, -2.4930336, -1.1625206, -1.0525147, -2.2532892, -1.5777700, -1.0525147, -1.9450794, -1.9450800, -1.0525147, -1.5777698, -2.2532895, -1.0525147, -1.1625204, -2.4930336, -1.0525147, -0.7119485, -2.6570284, -1.0525147, -0.2397443, -2.7402909, -1.0525147, 0.2397444, -2.7402909, -1.0525147, 0.7119485, -2.6570284, -1.0525147, 1.1625204, -2.4930336, -1.0525147, 1.5777698, -2.2532895, -1.0525147, 1.9450794, -1.9450800, -1.0525147, 2.2532890, -1.5777705, -1.0525147, 2.4930334, -1.1625211, -1.0525147, 2.6570284, -0.7119480, -1.0525147, 2.7402909, -0.2397438, -1.0525147, 2.9138803, 0.2183652, -0.3688816, 2.8487892, 0.6502175, -0.3688816, 2.7200608, 1.0675452, -0.3688816, 2.5305705, 1.4610256, -0.3688816, 2.2845514, 1.8218693, -0.3688816, 1.9874996, 2.1420152, -0.3688816, 1.6460500, 2.4143121, -0.3688816, 1.2678305, 2.6326771, -0.3688816, 0.8612897, 2.7922325, -0.3688816, 0.4355091, 2.8894143, -0.3688816, 0.0000002, 2.9220512, -0.3688816, -0.4355093, 2.8894143, -0.3688816, -0.8612896, 2.7922328, -0.3688816, -1.2678308, 2.6326771, -0.3688816, -1.6460500, 2.4143119, -0.3688816, -1.9875000, 2.1420147, -0.3688816, -2.2845516, 1.8218689, -0.3688816, -2.5305705, 1.4610257, -0.3688816, -2.7200606, 1.0675455, -0.3688816, -2.8487892, 0.6502175, -0.3688816, -2.9138806, 0.2183648, -0.3688816, -2.9138803, -0.2183653, -0.3688816, -2.8487892, -0.6502174, -0.3688816, -2.7200608, -1.0675448, -0.3688816, -2.5305705, -1.4610255, -0.3688816, -2.2845514, -1.8218695, -0.3688816, -1.9874996, -2.1420152, -0.3688816, -1.6460491, -2.4143126, -0.3688816, -1.2678310, -2.6326768, -0.3688816, -0.8612891, -2.7922328, -0.3688816, -0.4355102, -2.8894141, -0.3688816, 0.0000000, -2.9220512, -0.3688816, 0.4355102, -2.8894141, -0.3688816, 0.8612893, -2.7922328, -0.3688816, 1.2678310, -2.6326768, -0.3688816, 1.6460491, -2.4143126, -0.3688816, 1.9874996, -2.1420152, -0.3688816, 2.2845523, -1.8218682, -0.3688816, 2.5305703, -1.4610261, -0.3688816, 2.7200608, -1.0675447, -0.3688816, 2.8487890, -0.6502187, -0.3688816, 2.9138803, -0.2183652, -0.3688816}; 
