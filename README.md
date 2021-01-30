# text-restyle

## About
This repo contains my implementation of paper [Multiple-Attribute Text Style Transfer](https://arxiv.org/abs/1811.00552). It was done completely by me, in short time and basically as a programming challenge, so it may contain _a lot of bugs_. If you found one (and you probably will, if you try), please feel free to contact me/open an issue.

## Usage
At first, clone the repo.
```
git clone https://github.com/clock-me/text-restyle.git
cd text-restyle
```
Your data must be contained in `train.tsv`, `val.tsv`, and `test.tsv` for train set, validation set and test set respectively (names for files are arbitrary). All the data must be already preprocessed (but not necessarily tokenized). Examples of data may be found in folder `data_examples`. Also train configuration must be stored in `yaml` file. Example of such is `config.yaml`.
To run training one can use `train.py`:
```
train.py [-h] [--do_preprocess]
         path_to_config path_to_train path_to_val path_to_test
```
Where `do_preprocess` stands for "whether create files for subword tokenization or not". This flag must be raised in the first run.

## TODO
- [ ] Add testing utilities (Metrics such as self-bleu (which is arguably the worst in text-style-transfer), style accuraccy)
- [ ] Add complex style support (in this case, style is a vector, as in the paper)
- [ ] Reproduce paper results

## Work examples
The model with trained with hyperparameters, described `config.yaml`. I used the _yelp_ data, preprocessed & provided by https://github.com/shentianxiao/language-style-transfer. Here are some examples, generated by model (code is availible at `sandbox.ipynb`). These are generated with greedy decoding.

**From positive sentiment to negative:**
```
from: love this place !
to: do not go here !

from: the staff are always very nice and helpful .
to: the staff are always very unprofessional and helpful .

from: the new yorker was amazing .
to: the new yorker was terrible .

from: very ny style italian deli .
to: very ny style italian sub .

from: they have great sandwiches and homemade cookies .
to: they have no sandwiches and homemade tortillas .

from: great choice -- -- i 'll be back !
to: no choice -- -- i 'll be back !

from: tried their lasagna and its the best ive ever had .
to: tried their lasagna and its the worst ive ever had .

from: the food was amazing !
to: the food was terrible !

from: authentic new york and italy style .
to: mediocre and healthy italian .

from: cannoli were amazing .
to: the cashiers were terrible .

from: the owner is a very welcoming person , make sure to talk to him .
to: the owner is a very personable , make sure to talk to him .

from: this place is a gem .
to: this place is a nightmare .

from: eat in and take home everything .
to: eat in and take home everything .

from: and the staff is extremely friendly .
to: and the staff is extremely rude .

from: the sandwiches are huge and delicious !
to: the sandwiches are huge and gross !

from: cookie heaven !
to: eh !

from: best pastrami sandwich in town by far !
to: worst pastrami sandwich in town by far !

from: also lots of groceries to take home with you .
to: also lots of dirty to take home with you .

from: great people great place !
to: no people no thanks !
```

**From negative sentiment to positive:**
```
from: ok never going back to this place again .
to: always delicious going back to this place again .

from: easter day nothing open , heard about this place figured it would ok .
to: great daycare nothing , opened about this placed it would delicious .

from: the host that walked us to the table and left without a word .
to: the gentleman that we used to the table and left without a word .

from: it just gets worse .
to: it just gets best .

from: the food tasted awful .
to: the food tasted fantastic .

from: no sign of the manager .
to: great view of the manager .

from: the last couple years this place has been going down hill .
to: the last years this place has been going down hill .

from: last night however it was way to thick and tasteless .
to: last night however it was way to fresh and delicious .

from: it smelled like rotten urine .
to: it smelled like best chinese .

from: i am not exaggerating .
to: i am definitely recommend .

from: this smelled bad !
to: this smelled great !

from: it was obvious it was the same damn one he brought the first time .
to: it was obviously the same it was one brought the first time .

from: i tried to eat it but it was disgusting .
to: i tried to eat it but it was perfect .

from: it tasted horrible !
to: it tasted wonderful !

from: i pushed it aside and did n't eat anymore .
to: i enjoyed it ai and it did n't eat anymore .

from: i will never be back .
to: i will always be back .

from: do yourself a favor and just stay away .
to: do yourself a favor and just stay away .

from: i ordered a chicken sandwich with onion rings and a soda .
to: i ordered a chicken sandwich with onions and ahi .

from: $ _num_ for a soda ?
to: $ _num_ for a wedding !

from: the total for this lunch was $ _num_ .
to: the total for this lunch was $ _num_ .
```

## References
1) https://arxiv.org/abs/1811.00552
2) https://github.com/shentianxiao/language-style-transfer (data source)
