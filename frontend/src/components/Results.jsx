import React from "react";
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

function Results(props) {
  let results = Object.values(props.results);

  let actual = results[0];
  let predicted = results[1];

  let normalizedPredicted = ((predicted + 1) / 2) * 100

  return (
    <div>
      <h1>Results</h1>
      <h2>Predicted: {predicted} </h2>
      <div style={{ width: '500px', height: '500px', margin: '0 auto' }}>
        <CircularProgressbar 
          value={normalizedPredicted} 
          text={`${normalizedPredicted.toFixed(2)}%`} 
          styles={buildStyles({
            textColor: "#000000",
            pathColor: "#22d625",
            trailColor: "#de3a3a",
          })}
        />
      </div>
    </div>
  )

}

export default Results;
