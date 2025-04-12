// interactions.js - Minimal JavaScript for enhanced interactions

document.querySelector('.chat-submit-btn').addEventListener('click', sendMessage);

document.addEventListener('DOMContentLoaded', () => {
    // Function to handle parallax effects
    const handleParallax = () => {
      const parallaxElements = document.querySelectorAll('.parallax-element');
      
      parallaxElements.forEach(element => {
        const speed = element.getAttribute('data-speed') || 0.2;
        const yPos = -(window.scrollY * speed);
        element.style.transform = `translateY(${yPos}px)`;
      });
    };
  
    // Typing animation for AI messages
    const simulateTyping = (element, text, speed = 30) => {
      let i = 0;
      element.textContent = '';
      
      const typeNextChar = () => {
        if (i < text.length) {
          element.textContent += text.charAt(i);
          i++;
          setTimeout(typeNextChar, speed);
        }
      };
      
      typeNextChar();
    };
  
    // Initialize message sending functionality
    const initChat = () => {
      const chatInput = document.querySelector('.chat-input');
      const sendButton = document.querySelector('.send-button');
      const chatMessages = document.querySelector('.chat-messages');
      
      // Only initialize if we have the chat elements
      if (!chatInput || !chatMessages) return;
      
      const sendMessage = () => {
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Create user message
        const userMsg = document.createElement('div');
        userMsg.className = 'message message-user';
        userMsg.textContent = message;
        chatMessages.appendChild(userMsg);
        
        // Clear input
        chatInput.value = '';
        
        // Auto-scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // In a real app, you'd send this to an API
        // For now, we'll just simulate a response
        setTimeout(() => {
          const aiMsg = document.createElement('div');
          aiMsg.className = 'message message-ai';
          chatMessages.appendChild(aiMsg);
          
          // Simulate typing
          simulateTyping(aiMsg, "I appreciate your message. I'm just a demo AI, but in a real implementation, this would be connected to a backend service.");
          
          // Auto-scroll as the message "types"
          const scrollInterval = setInterval(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
          }, 100);
          
          // Clear interval after typing finishes
          setTimeout(() => clearInterval(scrollInterval), 3000);
        }, 1000);
      };
      
      // Send on button click
      if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
      }
      
      // Send on Enter key
      chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          sendMessage();
          e.preventDefault();
        }
      });
    };
  
    // Add ripple effect to buttons
    const addRippleEffect = () => {
      const buttons = document.querySelectorAll('.btn');
      
      buttons.forEach(button => {
        button.addEventListener('click', function(e) {
          const rect = this.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          
          const ripple = document.createElement('span');
          ripple.className = 'ripple';
          ripple.style.left = `${x}px`;
          ripple.style.top = `${y}px`;
          
          this.appendChild(ripple);
          
          setTimeout(() => {
            ripple.remove();
          }, 600);
        });
      });
    };
  
    // Initialize all interactions
    window.addEventListener('scroll', handleParallax);
    initChat();
    addRippleEffect();
  });
  