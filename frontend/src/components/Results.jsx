import React from "react";
import GaugeChart from 'react-gauge-chart';

function Results(props) {
  let results = Object.values(props.results);

  let actual = results[0];
  let predicted = results[1];

  let normalizedPredicted = (predicted + 1) / 2;

  return (
    <div>
      <h1>Results</h1>
      <h2>Predicted Score: {predicted} </h2>
      <GaugeChart
        id="gauge-chart"
        percent={normalizedPredicted}
        nrOfLevels={20}
        textColor="#000000"
        needleColor="#000000"
        colors={["#FF0000", "#F9C802", "#8BC34A"]}
      />
    </div>
  );
}

export default Results;
