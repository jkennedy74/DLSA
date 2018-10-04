// Select the submit buttons
var predict = d3.select("#predict");
var sentiment = d3.select("#sentiment");

function predictReturn() {
// Prevent the page from refreshing
  d3.event.preventDefault();

  // Print a log to the console
  console.log("Prediction analysis");

  var sampleText_url = '/sample_text';
  var seed_url = '/seed';
  var predText_url = '/lstm_output';
  
  d3.json(sampleText_url).then(function(data) {

    console.log(data[0]);

    var sampleText = d3.select("#sampleText");
    sampleText.html("");
    sampleText.text(data[0]);
  });

  d3.json(seed_url).then(function(data) {

    console.log(data[0]);

    var modelSeed = d3.select("#modelSeed");
    modelSeed.html("");
    modelSeed.text(data[0]);

  });

  d3.json(predText_url).then(function(data) {

    console.log(data[0]);

    var predictedText = d3.select("#predictedText");
    predictedText.html("");
    predictedText.text(data[0]);
  });
}

function sentimentReturn() {
    // Prevent the page from refreshing
  d3.event.preventDefault();

  // Print a log to the console
  console.log("Sentiment analysis");

  var rawResult_url = '/raw_result';
  var predResult_url = '/pred_result';

  var modelSentiment = d3.select("#modelSentiment");
  modelSentiment.html("");
  modelTable = modelSentiment.append("table");
  modelTable.classed("table table-striped table-responsive", true)
  modelThead = modelTable.append("thead");
  modelTbody = modelTable.append("tbody");

  var predictSentiment = d3.select("#predictSentiment");
  predictSentiment.html("");
  predictTable = predictSentiment.append("table");
  predictTable.classed("table table-striped table-responsive", true)
  predictThead = predictTable.append("thead");
  predictTbody = predictTable.append("tbody");

  d3.json(rawResult_url).then(function(data) {

    console.log(data);

    data.forEach((entry) => {
        var row = modelTbody.append("tr");
        Object.entries(entry).forEach(([key, value]) => {
          var cell = row.append("td");
          var desc = modelThead.append("th");
          desc.text(key);
          cell.text(value);
        });
  });
});

  d3.json(predResult_url).then(function(data) {

    console.log(data)

    data.forEach((entry) => {
        var row = predictTbody.append("tr");
        Object.entries(entry).forEach(([key, value]) => {
          var cell = row.append("td");
          var desc = predictThead.append("th");
          desc.text(key);
          cell.text(value);
        });
  });
});

}

// run the function above when the Search button is clicked
predict.on("click", predictReturn);
sentiment.on("click", sentimentReturn);