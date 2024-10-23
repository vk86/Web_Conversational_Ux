import React from 'react';
import './Styles.css'; // Import your styles
/*
const ResizablePosterBox = ({ id, title, description }) => {
  return (
    <div 
      className="resizable-box" 
      style={{ resize: 'both', overflow: 'auto', minWidth: '150px', minHeight: '100px' }}>
      <div className="id-box">{id}</div>
      <div className="poster-title">{title}</div>
      <div className="poster-description">{description}</div>
    </div>
    
  );
};
*/
const ResizablePosterBox = ({ id, title, description }) => {
  return (
    <div className="resizable-box">
      <div className="id-box-wrapper">
        <div className="id-box">{id}</div>
      </div>
      <div className="poster-title">{title}</div>
      <div className="poster-description">{description}</div>
    </div>
  );
};

export default ResizablePosterBox;




