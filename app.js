import './App.css';
import React, { useState, useEffect, useRef } from 'react';
import { w3cwebsocket as W3CWebSocket } from 'websocket';

const client = new W3CWebSocket('ws://localhost:8080');

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [application, setApplication] = useState('');
  const [feature, setFeature] = useState('');
  const [storyName, setStoryName] = useState('');
  const [storyDescription, setStoryDescription] = useState('');
  const [storyInput, setStoryInput] = useState('');
  const messageEndRef = useRef(null);

  useEffect(() => {
    client.onopen = () => {
      console.log('WebSocket Client Connected');
    };

    client.onmessage = (message) => {
      setMessages((prevMessages) => [...prevMessages, { sender: 'bot', text: message.data }]);
    };
  }, []);

  const sendMessage = () => {
    let message = "";
    if (application && feature && storyName.trim() && storyDescription.trim() && storyInput.trim()) {
      message = JSON.stringify({ storyName, storyDescription, storyInput, input });
    } else if (input.trim()) {
      message = input.trim();
    }

    if (message) {
      client.send(message);
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: 'me', text: `${input}\n ${storyName ? ('Story Name: ' + storyName + '\n Story Description: ' + storyDescription) : ''}` }
      ]);

      setInput('');
      setStoryName('');
      setStoryDescription('');
      setStoryInput('');
      setApplication('');
      setFeature('');
    }
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  const handleApplicationChange = (e) => {
    setApplication(e.target.value);
  };

  const handleFeatureChange = (e) => {
    setFeature(e.target.value);
  };

  const handleStoryNameChange = (e) => {
    setStoryName(e.target.value);
  };

  const handleStoryDescriptionChange = (e) => {
    setStoryDescription(e.target.value);
  };

  const handleStoryInputChange = (e) => {
    setStoryInput(e.target.value);
  };

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className='main-container'>
      <div className='header pannel'>Header</div>
      <div className="chat-container">
        <div className="left-column pannel">
          <select value={application} onChange={handleApplicationChange} className="dropdown">
            <option value="" disabled>Select Application</option>
            <option value="Project A">Project A</option>
            <option value="Project B">Project B</option>
          </select>
          <select value={feature} onChange={handleFeatureChange} className="dropdown">
            <option value="" disabled>Select Feature</option>
            <option value="User Story">User Story</option>
          </select>
        </div>
        <div className="middle-column pannel">
          <div className="chat-box">
            {messages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.sender}`}>
                {msg.text}
              </div>
            ))}
            <div ref={messageEndRef} />
          </div>
          {application && feature && (
            <div className="chat-story-container">
              <div className="story-form">
                <input
                  type="text"
                  placeholder="User Story Name"
                  value={storyName}
                  onChange={handleStoryNameChange}
                  className="story-input"
                />
                <input
                  type="text"
                  placeholder="Story Description"
                  value={storyDescription}
                  onChange={handleStoryDescriptionChange}
                  className="story-input"
                />
                <input
                  type="text"
                  placeholder="Story Input"
                  value={storyInput}
                  onChange={handleStoryInputChange}
                  className="story-input"
                />
              </div>
            </div>
          )}
          <div className="chat-input-container">
            <input
              type="text" 
              placeholder="Type your message..."
              value={input}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              className="chat-input"
            />
            <button
              onClick={sendMessage}
              className="chat-send-button"
              disabled={!input.trim() && !(storyName.trim() && storyDescription.trim() && storyInput.trim())}
            >
              Send
            </button>
          </div>
        </div>
        <div className="right-column pannel">

        </div>
      </div>
    </div>
  );
}

export default App;
