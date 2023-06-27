import React, { useState } from 'react';
import axios from 'axios';

function SearchBar(props) {
  const [company, setCompany] = useState('');
  const [predictType, setPredictType] = useState('');
  const [predictPeriod, setPredictPeriod] = useState('');

  const handleSearch = async () => {
    try {
      const response = await axios.post('http://localhost:8080/predict', {
        company: company,
        predict_type: predictType,
        predict_period: predictPeriod
      });
      props.onSearchResults(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <label htmlFor="company">Company:</label>
      <input type="text" id="company" value={company} onChange={(event) => setCompany(event.target.value)} />
      <label htmlFor="predictType">Prediction Type:</label>
      <select id="predictType" value={predictType} onChange={(event) => setPredictType(event.target.value)}>
        <option value="">Select a type</option>
        <option value="Day">Day</option>
        <option value="Week">Week</option>
        <option value="Month">Month</option>
      </select>
      <label htmlFor="predictPeriod">Prediction Period:</label>
      <input type="number" id="predictPeriod" value={predictPeriod} onChange={(event) => setPredictPeriod(event.target.value)} />
      <button onClick={handleSearch}>Search</button>
    </div>
  );
}

export default SearchBar;