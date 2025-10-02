// src/mockData.js

export const mockSuccessData = {
  sentiment_report: `
### Overall Sentiment: **Mixed to Slightly Positive** 

Public opinion on the new **"Quantum Leap Initiative"** is largely divided. There's significant excitement from the tech and science communities, viewing it as a major step forward. However, there is also notable skepticism from economic analysts regarding the project's massive budget and uncertain immediate returns.

**Key Themes:**
- **Positive:** Technological advancement, national prestige, future potential.
- **Negative:** High cost, potential for misuse, long timeline for results.

**Examples:**
1.  **(Positive)** _"Incredible news! The Quantum Leap Initiative puts our country at the forefront of the next technological revolution. Can't wait to see the breakthroughs."_ - @TechForward on X
2.  **(Negative)** _"Another billion-dollar project with no clear ROI. How about we fix our roads first before we try to build a quantum computer?"_ - Comment on a news article
`,
  reasoning_report: `
### Why is Sentiment Mixed?

The division in public sentiment can be traced to a classic **"long-term vision vs. immediate needs"** conflict.

1.  **Tech Optimism:** The positive sentiment is fueled by a narrative of progress and innovation. News articles and experts highlighted in Wikipedia articles emphasize the potential for quantum computing to solve major global problems, from medicine to climate change. This creates a strong "in-group" of supporters who are technologically literate.

2.  **Economic Pragmatism:** The negative sentiment stems from tangible concerns about resource allocation. Social media discussions often pivot to the opportunity cost—what other public services could be funded with the initiative's budget? This view is amplified by news reports focusing on the economic and logistical challenges of the project. The lack of immediate, visible benefits makes it a hard sell for the general public concerned with day-to-day issues.
`,
  rag1_results: [
    { id: 'abc1', source: 'youtube', sentiment: 'positive', text: "This is a giant leap for science! So proud of the researchers and engineers making this happen. The future is now!", link: '#' },
    { id: 'def2', source: 'reddit', sentiment: 'negative', text: "I'm extremely skeptical. We've heard these grand promises before. I'll believe it when I see tangible results, not just press releases.", link: '#' },
    { id: 'ghi3', source: 'reddit', sentiment: 'neutral', text: "Interesting development, but it's too early to say if it will succeed. The technical challenges are immense.", link: '#' },
    { id: 'jkl4', source: 'youtube', sentiment: 'positive', text: "Amazing! The potential applications for medicine and materials science are mind-boggling. A worthy investment.", link: '#' },
  ],
  rag2_results: [
    { url: '#', title: "Govt Announces Ambitious 'Quantum Leap Initiative'", content: "Officials today unveiled a landmark multi-billion dollar project aimed at achieving quantum supremacy within the next decade..." },
    { url: '#', title: "Wikipedia: Quantum Computing", content: "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement..." },
    { url: '#', title: "Economists Question High Cost of Quantum Project", content: "While the scientific community celebrates, some economists are raising red flags over the initiative's hefty price tag and the long road to commercial viability..." },
  ],
};

export const mockErrorData = "Failed to analyze the topic. The model may be overloaded. Please try again later.";