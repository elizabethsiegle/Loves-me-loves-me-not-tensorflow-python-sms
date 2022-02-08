### Answer the "loves me, loves me not" question with Twilio and TensorFlow.

Complete tutorial can be found [here on the Twilio blog](https://www.twilio.com/blog/classify-texts-with-tensorflow-and-twilio-to-answer-loves-me-loves-me-not).

To run, install <em>requirements.txt</em> with `pip3 install -r requirements.txt` then run the Flask app with 
```
export FLASK_APP=main

export FLASK_ENV=development

flask run  --without-threads
```
Please ignore some of the commits and commit messages, it was a pain to deploy to Heroku because the TensorFlow model was computationally-intensive.

![](https://countrush-prod.azurewebsites.net/l/badge/?repository=elizabethsiegle.Loves-me-loves-me-not-tensorflow-python-sms)
