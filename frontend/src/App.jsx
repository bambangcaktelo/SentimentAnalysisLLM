import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

// --- API Configuration ---
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
const DAILY_LIMIT = 5;

// --- Centralized Axios Instance ---
const api = axios.create({
    baseURL: API_BASE_URL
});

api.interceptors.request.use(config => {
    const token = localStorage.getItem('authToken');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// =================================================================================
// --- HELPER & UI COMPONENTS ---
// =================================================================================

const LoadingSpinner = ({ text }) => (
    <div className="flex flex-col items-center justify-center space-y-4 my-12 animate-fadeIn">
        <div className="relative">
            <div className="w-20 h-20 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
            <div className="absolute inset-0 w-20 h-20 border-4 border-transparent border-r-purple-600 rounded-full animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}></div>
        </div>
        <p className="text-lg text-gray-300 animate-pulse">{text}</p>
    </div>
);

const ReportSection = ({ title, content }) => (
    <div className="bg-gradient-to-br from-gray-800 to-gray-850 p-6 md:p-8 rounded-2xl shadow-2xl border border-gray-700 hover:border-blue-500 transition-all duration-300 transform hover:scale-[1.01] animate-fadeIn">
        <div className="flex items-center mb-4 space-x-3">
            <div className="w-1 h-8 bg-gradient-to-b from-blue-500 to-purple-600 rounded-full"></div>
            <h2 className="text-2xl md:text-3xl font-bold text-white">{title}</h2>
        </div>
        <div className="prose prose-invert prose-lg max-w-none">
            <ReactMarkdown>{content || "No content available."}</ReactMarkdown>
        </div>
    </div>
);

const QuantitativeSection = ({ data }) => {
    if (!data || !data.distribution) return null;
    const { distribution, top_words, word_cloud } = data;
    const total = Object.values(distribution).reduce((sum, val) => sum + val, 0) || 1;

    return (
        <div className="bg-gradient-to-br from-gray-800 to-gray-850 p-6 md:p-8 rounded-2xl shadow-2xl border border-gray-700 animate-fadeIn">
            <div className="flex items-center mb-6 space-x-3">
                <div className="w-1 h-8 bg-gradient-to-b from-blue-500 to-purple-600 rounded-full"></div>
                <h2 className="text-2xl md:text-3xl font-bold text-white">Quantitative Analysis</h2>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="space-y-6">
                    <h3 className="text-xl font-semibold text-blue-400 mb-4 flex items-center">
                        <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        Sentiment Distribution
                    </h3>
                    <div className="space-y-4">
                        {Object.entries(distribution).map(([sentiment, count]) => (
                            <div key={sentiment} className="transform hover:scale-105 transition-transform">
                                <div className="flex justify-between mb-2">
                                    <span className="text-base font-semibold text-gray-200 capitalize">{sentiment}</span>
                                    <span className="text-sm font-bold text-gray-300">{count} posts ({((count / total) * 100).toFixed(1)}%)</span>
                                </div>
                                <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden shadow-inner">
                                    <div 
                                        className={`h-3 rounded-full transition-all duration-1000 ease-out ${
                                            sentiment === 'positive' ? 'bg-gradient-to-r from-green-500 to-green-400' : 
                                            sentiment === 'negative' ? 'bg-gradient-to-r from-red-500 to-red-400' : 
                                            'bg-gradient-to-r from-gray-500 to-gray-400'
                                        }`} 
                                        style={{ width: `${(count / total) * 100}%` }}
                                    ></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
                <div>
                    <h3 className="text-xl font-semibold text-purple-400 mb-4 flex items-center">
                        <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                        </svg>
                        Top Keywords
                    </h3>
                    <ul className="space-y-2">
                        {top_words && top_words.map(([word, count], idx) => (
                            <li key={word} className="flex justify-between items-center bg-gradient-to-r from-gray-700 to-gray-750 p-3 rounded-lg hover:from-gray-650 hover:to-gray-700 transition-all transform hover:translate-x-2 animate-fadeIn" style={{ animationDelay: `${idx * 50}ms` }}>
                                <span className="text-gray-200 font-medium">{word}</span>
                                <span className="text-sm font-bold text-purple-400 bg-purple-900/30 px-3 py-1 rounded-full">{count}</span>
                            </li>
                        ))}
                    </ul>
                </div>
                <div className="lg:col-span-2">
                    <h3 className="text-xl font-semibold text-indigo-400 mb-4 flex items-center">
                        <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                        </svg>
                        Word Cloud
                    </h3>
                    {word_cloud ? (
                        <div className="bg-white rounded-xl shadow-lg overflow-hidden transform hover:scale-[1.02] transition-transform">
                            <img src={word_cloud} alt="Word Cloud" className="w-full" />
                        </div>
                    ) : (
                        <p className="text-gray-400 text-center p-8 bg-gray-750 rounded-xl">Not enough data to generate a word cloud.</p>
                    )}
                </div>
            </div>
        </div>
    );
};

const DocumentCard = ({ doc }) => (
    <div className="bg-gradient-to-br from-gray-800 to-gray-850 p-5 rounded-xl shadow-lg border border-gray-700 hover:border-blue-500 transition-all duration-300 transform hover:scale-105 hover:shadow-2xl animate-fadeIn">
        <div className="flex justify-between items-start mb-3">
            <span className="text-sm font-bold text-blue-400 capitalize bg-blue-900/30 px-3 py-1 rounded-full">{doc.source}</span>
            {doc.sentiment && (
                <span className={`text-xs font-bold px-3 py-1 rounded-full shadow-md ${
                    doc.sentiment === 'positive' ? 'bg-gradient-to-r from-green-500 to-green-600' : 
                    doc.sentiment === 'negative' ? 'bg-gradient-to-r from-red-500 to-red-600' : 
                    'bg-gradient-to-r from-gray-500 to-gray-600'
                } text-white`}>
                    {doc.sentiment}
                </span>
            )}
        </div>
        {doc.content ? (
            <div className="text-gray-300 text-sm mb-4 line-clamp-4" dangerouslySetInnerHTML={{ __html: doc.content.slice(0, 200) + '...' }} />
        ) : (
            <p className="text-gray-300 text-sm mb-4 line-clamp-4">"{doc.text?.slice(0, 200)}..."</p>
        )}
        <a href={doc.link || doc.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center text-blue-400 hover:text-blue-300 text-sm font-semibold transition-colors group">
            View Source
            <svg className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
        </a>
    </div>
);

const ChatWindow = ({ messages, onSendMessage, isLoading, isReasoningGenerating, questionsRemaining, token }) => {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef(null);

    useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

    const handleSend = (e) => {
        e.preventDefault();
        if (input.trim() && !isLoading && !isReasoningGenerating && (questionsRemaining > 0 || !token)) {
            onSendMessage(input);
            setInput('');
        }
    };
    
    if (!token) {
        return (
            <div className="bg-gradient-to-br from-gray-800 to-gray-850 rounded-2xl shadow-2xl border border-gray-700 mt-10 text-center p-8 md:p-12 animate-fadeIn">
                <div className="max-w-md mx-auto">
                    <svg className="w-20 h-20 mx-auto mb-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    <h2 className="text-3xl font-bold text-white mb-3">Unlock AI Insights</h2>
                    <p className="text-gray-400 mb-6 text-lg">Log in to ask follow-up questions and dive deeper into the political analysis.</p>
                    <button 
                        onClick={() => window.dispatchEvent(new CustomEvent('show-auth'))} 
                        className="bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-3 px-8 rounded-xl hover:from-blue-700 hover:to-purple-700 transform hover:scale-105 transition-all shadow-lg"
                    >
                        Login to Chat
                    </button>
                </div>
            </div>
        );
    }
    
    return (
        <div className="bg-gradient-to-br from-gray-800 to-gray-850 rounded-2xl shadow-2xl border border-gray-700 mt-10 overflow-hidden animate-fadeIn">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-4 md:p-6">
                <h2 className="text-2xl md:text-3xl font-bold text-white flex items-center">
                    <svg className="w-8 h-8 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                    </svg>
                    Ask About This Analysis
                </h2>
                <p className="text-sm text-blue-100 mt-2 flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    {questionsRemaining} questions remaining today
                </p>
            </div>
            <div className="p-4 md:p-6 h-96 overflow-y-auto flex flex-col space-y-4 bg-gray-900/50">
                {messages.map((msg, index) => (
                    <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-slideIn`}>
                        <div className={`max-w-[85%] md:max-w-lg p-4 rounded-2xl shadow-lg ${
                            msg.role === 'user' 
                                ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white' 
                                : 'bg-gradient-to-r from-gray-700 to-gray-750 text-gray-100'
                        }`}>
                            <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                    </div>
                ))}
                {isReasoningGenerating && (
                    <div className="text-center p-6">
                        <div className="inline-flex items-center space-x-2 text-blue-400">
                            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                            <p className="ml-3 animate-pulse">Generating initial reasoning...</p>
                        </div>
                    </div>
                )}
                {isLoading && (
                    <div className="flex justify-start animate-slideIn">
                        <div className="bg-gradient-to-r from-gray-700 to-gray-750 text-gray-200 p-4 rounded-2xl shadow-lg">
                            <span className="inline-flex items-center space-x-2">
                                <span className="animate-pulse">Typing</span>
                                <span className="flex space-x-1">
                                    <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                    <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                    <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                                </span>
                            </span>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>
            <form onSubmit={handleSend} className="p-4 md:p-6 bg-gray-900/70 border-t border-gray-700">
                <div className="flex gap-3">
                    <input 
                        type="text" 
                        value={input} 
                        onChange={(e) => setInput(e.target.value)} 
                        placeholder={questionsRemaining > 0 ? "Ask a follow-up question..." : "You have no questions left today."} 
                        className="flex-grow bg-gray-800 text-white rounded-xl px-5 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 border border-gray-700 transition-all" 
                        disabled={isLoading || isReasoningGenerating || questionsRemaining <= 0} 
                    />
                    <button 
                        type="submit" 
                        className="bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-3 px-6 md:px-8 rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 transition-all shadow-lg flex items-center" 
                        disabled={isLoading || isReasoningGenerating || !input.trim() || questionsRemaining <= 0}
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                        </svg>
                    </button>
                </div>
            </form>
        </div>
    );
};

// =================================================================================
// --- PAGE COMPONENTS ---
// =================================================================================

const AnalysisPage = ({ token }) => {
    const [query, setQuery] = useState('');
    const [jobId, setJobId] = useState(localStorage.getItem('jobId'));
    const [status, setStatus] = useState('idle');
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [messages, setMessages] = useState([]);
    const [isChatLoading, setIsChatLoading] = useState(false);
    const [questionsRemaining, setQuestionsRemaining] = useState(0);
    const [isReasoningGenerating, setIsReasoningGenerating] = useState(false);
    const pollIntervalRef = useRef(null);

    const loadAnalysisFromHistory = (selectedJobId) => {
        setResult(null); setError(null); setMessages([]);
        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        setJobId(selectedJobId);
        setStatus('pending');
        localStorage.setItem('jobId', selectedJobId);
    };

    useEffect(() => {
        window.loadAnalysisFromHistory = loadAnalysisFromHistory;
        return () => delete window.loadAnalysisFromHistory;
    }, []);

    useEffect(() => {
        if (jobId && !result) setStatus('pending');
    }, [jobId]);

    useEffect(() => {
        const pollStatus = async () => {
            const statusEndpoint = token ? `/status/${jobId}` : `/guest-status/${jobId}`;
            try {
                const { data } = await api.get(statusEndpoint);
                setQuestionsRemaining(data.questions_remaining);

                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(pollIntervalRef.current);
                    setResult(data.result);
                    setStatus(data.status === 'completed' ? 'success' : 'error');
                    if (data.status === 'failed') {
                        setError(data.result?.error || 'Analysis failed.');
                    } else if (token) {
                        setMessages(data.result?.chat_history?.length ? data.result.chat_history : [{ role: 'assistant', content: "Analysis complete. Ask questions." }]);
                    }
                }
            } catch (err) {
                setError(err.response?.data?.detail || 'Could not retrieve analysis status.');
                setStatus('error');
                clearInterval(pollIntervalRef.current);
            }
        };

        if (status === 'pending' && jobId) {
            pollStatus();
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = setInterval(pollStatus, 3000);
        }
        return () => clearInterval(pollIntervalRef.current);
    }, [status, jobId, token]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!query.trim() || status === 'pending') return;
        setResult(null); setError(null); setJobId(null); setMessages([]);
        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        
        setStatus('pending');
        const endpoint = token ? '/analyze' : '/guest-analyze';

        try {
            const response = await api.post(endpoint, { query });
            const newJobId = response.data.job_id;
            setJobId(newJobId);
            localStorage.setItem('jobId', newJobId);
            if(token) window.dispatchEvent(new CustomEvent('refresh-history'));
        } catch (err) { 
            setError(err.response?.data?.detail || 'Failed to start analysis.'); 
            setStatus('error'); 
        }
    };
    
    const handleSendMessage = async (userInput) => {
        if (!token) return;
        const newMessages = [...messages, { role: 'user', content: userInput }];
        setMessages(newMessages);
        const isFirstUserMessage = messages.filter(m => m.role === 'user').length === 0;

        if (isFirstUserMessage) setIsReasoningGenerating(true); 
        else setIsChatLoading(true);

        try {
            if (isFirstUserMessage) await api.post(`/generate-reasoning/${jobId}`);
            const response = await api.post(`/chat/${jobId}`, { messages: newMessages });
            setMessages([...newMessages, { role: 'assistant', content: response.data.response }]);
            setQuestionsRemaining(response.data.questions_remaining);
        } catch (err) {
            setMessages([...newMessages, { role: 'assistant', content: `Error: ${err.response?.data?.detail || "An error occurred."}` }]);
        } finally { 
            setIsChatLoading(false); 
            setIsReasoningGenerating(false); 
        }
    };

    return (
        <main className="animate-fadeIn">
            <form onSubmit={handleSubmit} className="mb-12">
                <div className="flex flex-col gap-4">
                    <div className="relative">
                        <input 
                            type="text" 
                            value={query} 
                            onChange={(e) => setQuery(e.target.value)} 
                            placeholder="e.g., Presidential election sentiment in swing states..." 
                            className="w-full bg-gray-800 text-white border-2 border-gray-700 rounded-xl px-5 py-4 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all" 
                            disabled={status === 'pending'} 
                        />
                        <svg className="absolute right-4 top-1/2 transform -translate-y-1/2 w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                    </div>
                    <button 
                        type="submit" 
                        className="bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-4 px-8 rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 transform hover:scale-105 transition-all shadow-lg flex items-center justify-center space-x-2" 
                        disabled={status === 'pending' || !query.trim()}
                    >
                        {status === 'pending' ? (
                            <>
                                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                <span>Analyzing...</span>
                            </>
                        ) : (
                            <>
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                                </svg>
                                <span>Analyze Sentiment</span>
                            </>
                        )}
                    </button>
                </div>
            </form>
            
            <div className="space-y-10">
                {status === 'pending' && <LoadingSpinner text="Analyzing political sentiment across multiple sources..." />}
                {status === 'error' && (
                    <div className="bg-gradient-to-r from-red-900/50 to-red-800/50 border border-red-700 p-6 rounded-xl text-center animate-shake">
                        <svg className="w-12 h-12 mx-auto mb-3 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <p className="text-red-300 text-lg">{error}</p>
                    </div>
                )}
                {status === 'success' && result && (
                    <div className="space-y-10">
                        <ReportSection title="Political Sentiment Report" content={result.sentiment_report} />
                        <QuantitativeSection data={result.quantitative_analysis} />
                        
                        <div className="animate-fadeIn">
                            <div className="flex items-center mb-6 space-x-3">
                                <div className="w-1 h-8 bg-gradient-to-b from-blue-500 to-purple-600 rounded-full"></div>
                                <h2 className="text-2xl md:text-3xl font-bold text-white">Social Media Pulse</h2>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {result.rag1_results?.map((doc, index) => <DocumentCard key={`rag1-${doc.id || index}`} doc={doc} />)}
                            </div>
                        </div>
                        
                        <div className="animate-fadeIn">
                            <div className="flex items-center mb-6 space-x-3">
                                <div className="w-1 h-8 bg-gradient-to-b from-blue-500 to-purple-600 rounded-full"></div>
                                <h2 className="text-2xl md:text-3xl font-bold text-white">News & Context</h2>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {result.rag2_results?.map((doc, index) => <DocumentCard key={`rag2-${doc.url || index}`} doc={doc} />)}
                            </div>
                        </div>
                        
                        <ChatWindow 
                            token={token}
                            messages={messages} 
                            onSendMessage={handleSendMessage} 
                            isLoading={isChatLoading} 
                            questionsRemaining={questionsRemaining} 
                            isReasoningGenerating={isReasoningGenerating} 
                        />
                    </div>
                )}
            </div>
        </main>
    );
};

const NotFoundPage = () => (
    <div className="text-center py-20 animate-fadeIn">
        <div className="max-w-2xl mx-auto">
            <div className="relative mb-8">
                <h1 className="text-9xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">404</h1>
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20 blur-3xl -z-10"></div>
            </div>
            <h2 className="text-3xl font-bold text-white mb-4">Page Not Found</h2>
            <p className="text-gray-400 text-lg mb-8">The political data you're looking for doesn't exist in this dimension.</p>
            <a 
                href="/" 
                className="inline-flex items-center bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-3 px-8 rounded-xl hover:from-blue-700 hover:to-purple-700 transform hover:scale-105 transition-all shadow-lg"
            >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                </svg>
                Return Home
            </a>
        </div>
    </div>
);

const VerificationPage = () => {
    const [message, setMessage] = useState('Verifying your email, please wait...');
    const [isSuccess, setIsSuccess] = useState(false);
    
    useEffect(() => {
        const verify = async () => {
            const token = new URLSearchParams(window.location.search).get('token');
            if (!token) {
                setMessage('Error: No verification token found in URL.');
                return;
            }
            try {
                const response = await api.post(`/verify-email`, null, { params: { token } });
                setMessage(response.data.message + " You will be redirected to the main page.");
                setIsSuccess(true);
                setTimeout(() => window.location.href = '/', 3000);
            } catch (error) {
                setMessage(error.response?.data?.detail || 'Verification failed. The link may be expired.');
            }
        };
        verify();
    }, []);

    return (
        <div className="text-center p-10 animate-fadeIn max-w-2xl mx-auto">
            <div className={`w-24 h-24 mx-auto mb-6 rounded-full flex items-center justify-center ${isSuccess ? 'bg-green-500/20' : 'bg-blue-500/20'}`}>
                {isSuccess ? (
                    <svg className="w-12 h-12 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                ) : (
                    <div className="w-12 h-12 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                )}
            </div>
            <h2 className="text-3xl font-bold mb-4 text-white">Account Verification</h2>
            <p className="text-gray-300 text-lg">{message}</p>
        </div>
    );
};

const ForgotPasswordPage = () => {
    const [email, setEmail] = useState('');
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setMessage(''); setError(''); setLoading(true);
        try {
            const response = await api.post(`/forgot-password`, { email });
            setMessage(response.data.message);
        } catch (err) {
            setError(err.response?.data?.detail || 'An error occurred.');
        } finally { setLoading(false); }
    };

    return (
        <div className="text-center p-6 md:p-10 max-w-lg mx-auto animate-fadeIn">
            <div className="bg-gradient-to-br from-gray-800 to-gray-850 p-8 rounded-2xl shadow-2xl border border-gray-700">
                <svg className="w-16 h-16 mx-auto mb-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
                </svg>
                <h2 className="text-3xl font-bold mb-4 text-white">Forgot Password</h2>
                <p className="text-gray-400 mb-6">Enter your account's email address and we'll send you a password reset link.</p>
                
                {message && (
                    <div className="bg-green-900/50 border border-green-700 text-green-300 p-4 rounded-xl mb-4 animate-slideIn">
                        <p>{message}</p>
                    </div>
                )}
                {error && (
                    <div className="bg-red-900/50 border border-red-700 text-red-400 p-4 rounded-xl mb-4 animate-shake">
                        <p>{error}</p>
                    </div>
                )}
                
                <form onSubmit={handleSubmit}>
                    <input 
                        id="email" 
                        type="email" 
                        value={email} 
                        onChange={(e) => setEmail(e.target.value)} 
                        placeholder="Enter your email" 
                        className="w-full bg-gray-700 text-white border-2 border-gray-600 rounded-xl px-4 py-3 mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all" 
                        required 
                    />
                    <button 
                        type="submit" 
                        disabled={loading} 
                        className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-3 rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 transform hover:scale-105 transition-all shadow-lg"
                    >
                        {loading ? 'Sending...' : 'Send Reset Link'}
                    </button>
                </form>
            </div>
        </div>
    );
};

const ResetPasswordPage = () => {
    const [password, setPassword] = useState('');
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const token = new URLSearchParams(window.location.search).get('token');

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!token) {
            setError('No reset token found. Please request a new link.');
            return;
        }
        setMessage(''); setError(''); setLoading(true);
        try {
            const response = await api.post(`/reset-password`, { token: token, new_password: password });
            setMessage(response.data.message + " You can now log in.");
            setTimeout(() => {
                window.location.href = '/';
                window.dispatchEvent(new CustomEvent('show-auth'));
            }, 3000);
        } catch (err) {
            setError(err.response?.data?.detail || 'An error occurred. The link may be expired.');
        } finally { setLoading(false); }
    };

    return (
        <div className="text-center p-6 md:p-10 max-w-lg mx-auto animate-fadeIn">
            <div className="bg-gradient-to-br from-gray-800 to-gray-850 p-8 rounded-2xl shadow-2xl border border-gray-700">
                <svg className="w-16 h-16 mx-auto mb-6 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
                <h2 className="text-3xl font-bold mb-4 text-white">Reset Your Password</h2>
                
                {message && (
                    <div className="bg-green-900/50 border border-green-700 text-green-300 p-4 rounded-xl mb-4 animate-slideIn">
                        <p>{message}</p>
                    </div>
                )}
                {error && (
                    <div className="bg-red-900/50 border border-red-700 text-red-400 p-4 rounded-xl mb-4 animate-shake">
                        <p>{error}</p>
                    </div>
                )}
                
                <form onSubmit={handleSubmit}>
                    <input 
                        type="password" 
                        value={password} 
                        onChange={(e) => setPassword(e.target.value)} 
                        placeholder="Enter your new password" 
                        className="w-full bg-gray-700 text-white border-2 border-gray-600 rounded-xl px-4 py-3 mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all" 
                        required 
                    />
                    <button 
                        type="submit" 
                        disabled={loading} 
                        className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-3 rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 transform hover:scale-105 transition-all shadow-lg"
                    >
                        {loading ? 'Resetting...' : 'Reset Password'}
                    </button>
                </form>
            </div>
        </div>
    );
};

// =================================================================================
// --- MODAL & HEADER COMPONENTS ---
// =================================================================================

const AuthModal = ({ onLoginSuccess, onClose }) => {
    const [isLoginView, setIsLoginView] = useState(true);
    const [error, setError] = useState('');
    const [message, setMessage] = useState('');
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleAuth = async (e) => {
        e.preventDefault();
        setError('');
        setMessage('');
        const endpoint = isLoginView ? '/token' : '/register';
        const payload = isLoginView ? { username, password } : { username, email, password };

        try {
            const response = await api.post(endpoint, payload);
            if (isLoginView) {
                const newToken = response.data.access_token;
                onLoginSuccess(newToken);
            } else {
                setMessage(response.data.message);
                setIsLoginView(true);
            }
        } catch (err) { setError(err.response?.data?.detail || 'An error occurred.'); }
    };

    return (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fadeIn">
            <div className="bg-gradient-to-br from-gray-800 to-gray-850 p-8 rounded-2xl shadow-2xl max-w-md w-full border border-gray-700 animate-scaleIn" onClick={(e) => e.stopPropagation()}>
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-3xl font-bold text-white">{isLoginView ? 'Welcome Back' : 'Join Us'}</h2>
                    <button 
                        onClick={onClose} 
                        className="text-gray-400 hover:text-white text-3xl transition-colors transform hover:rotate-90 duration-300"
                    >
                        &times;
                    </button>
                </div>
                
                <form onSubmit={handleAuth}>
                    {error && (
                        <div className="bg-red-900/50 border border-red-700 text-red-400 p-3 rounded-xl mb-4 animate-shake">
                            <p className="text-sm">{error}</p>
                        </div>
                    )}
                    {message && (
                        <div className="bg-blue-900/50 border border-blue-700 text-blue-300 p-3 rounded-xl mb-4 animate-slideIn">
                            <p className="text-sm">{message}</p>
                        </div>
                    )}
                    
                    <div className="space-y-4">
                        <div>
                            <label className="block text-gray-400 mb-2 font-semibold" htmlFor="username">Username</label>
                            <input 
                                id="username" 
                                type="text" 
                                value={username} 
                                onChange={(e) => setUsername(e.target.value)} 
                                className="w-full bg-gray-700 text-white border-2 border-gray-600 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all" 
                                required 
                            />
                        </div>
                        
                        {!isLoginView && (
                            <div>
                                <label className="block text-gray-400 mb-2 font-semibold" htmlFor="email">Email</label>
                                <input 
                                    id="email" 
                                    type="email" 
                                    value={email} 
                                    onChange={(e) => setEmail(e.target.value)} 
                                    className="w-full bg-gray-700 text-white border-2 border-gray-600 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all" 
                                    required 
                                />
                            </div>
                        )}
                        
                        <div>
                            <label className="block text-gray-400 mb-2 font-semibold" htmlFor="password">Password</label>
                            <input 
                                id="password" 
                                type="password" 
                                value={password} 
                                onChange={(e) => setPassword(e.target.value)} 
                                className="w-full bg-gray-700 text-white border-2 border-gray-600 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all" 
                                required 
                            />
                        </div>
                    </div>
                    
                    <button 
                        type="submit" 
                        className="w-full mt-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-3 rounded-xl hover:from-blue-700 hover:to-purple-700 transform hover:scale-105 transition-all shadow-lg"
                    >
                        {isLoginView ? 'Login' : 'Create Account'}
                    </button>
                    
                    <div className="text-center text-gray-400 mt-6">
                        {isLoginView ? "Don't have an account? " : "Already have an account? "}
                        <button 
                            type="button" 
                            onClick={() => setIsLoginView(!isLoginView)} 
                            className="text-blue-400 hover:text-blue-300 font-semibold hover:underline transition-colors"
                        >
                            {isLoginView ? 'Register here' : 'Login here'}
                        </button>
                    </div>
                    
                    {isLoginView && (
                        <div className="text-center mt-4">
                            <a href="/forgot-password" className="text-sm text-blue-400 hover:text-blue-300 hover:underline transition-colors">
                                Forgot Password?
                            </a>
                        </div>
                    )}
                </form>
            </div>
        </div>
    );
};

const HistoryPanel = ({ onClose, onClearAll, onDeleteItem, history }) => {
    return (
        <div className="fixed top-0 left-0 w-full sm:w-96 h-full bg-gradient-to-b from-gray-800 to-gray-900 shadow-2xl z-50 flex flex-col border-r border-gray-700 animate-slideInLeft">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6 flex justify-between items-center">
                <div className="flex items-center space-x-3">
                    <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <h2 className="text-2xl font-bold text-white">History</h2>
                </div>
                <button 
                    onClick={onClose} 
                    className="text-white hover:bg-white/20 rounded-full p-2 text-2xl transition-all transform hover:rotate-90"
                >
                    &times;
                </button>
            </div>
            
            <ul className="overflow-y-auto flex-grow p-4 space-y-3">
                {history.length > 0 ? history.map((item, idx) => (
                    <li key={item.job_id} className="group animate-fadeIn" style={{ animationDelay: `${idx * 50}ms` }}>
                        <div className="flex items-stretch gap-2">
                            <button 
                                onClick={() => { window.loadAnalysisFromHistory(item.job_id); onClose(); }} 
                                className="flex-grow text-left p-4 bg-gradient-to-br from-gray-700 to-gray-750 rounded-xl hover:from-gray-650 hover:to-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all transform hover:scale-[1.02] border border-gray-600 hover:border-blue-500"
                            >
                                <p className="text-white font-semibold truncate mb-1">{item.result.query}</p>
                                <p className="text-xs text-gray-400 flex items-center">
                                    <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    {new Date(item.created_at).toLocaleString()}
                                </p>
                            </button>
                            <button 
                                onClick={() => onDeleteItem(item.job_id)} 
                                className="px-3 bg-gray-700 hover:bg-red-600 text-gray-400 hover:text-white rounded-xl transition-all transform hover:scale-110 focus:outline-none focus:ring-2 focus:ring-red-500"
                                title="Delete"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                                </svg>
                            </button>
                        </div>
                    </li>
                )) : (
                    <div className="text-center py-16">
                        <svg className="w-20 h-20 mx-auto text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        <p className="text-gray-400">No history found.</p>
                        <p className="text-gray-500 text-sm mt-2">Your analyses will appear here</p>
                    </div>
                )}
            </ul>
            
            {history.length > 0 && (
                <div className="p-4 border-t border-gray-700 bg-gray-900">
                    <button 
                        onClick={onClearAll} 
                        className="w-full p-3 bg-gradient-to-r from-red-600 to-red-700 rounded-xl hover:from-red-700 hover:to-red-800 text-white font-semibold transform hover:scale-105 transition-all shadow-lg flex items-center justify-center space-x-2"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                        <span>Clear All History</span>
                    </button>
                </div>
            )}
        </div>
    );
};

const Header = ({ token, onLogout, onShowAuth, onShowHistory }) => (
    <header className="mb-12 animate-fadeIn">
        <nav className="flex justify-between items-center mb-8 flex-wrap gap-4">
            <div>
                {token && (
                    <button 
                        onClick={onShowHistory} 
                        className="bg-gray-800 hover:bg-gray-700 border border-gray-700 hover:border-blue-500 text-white font-semibold py-2.5 px-5 rounded-xl transition-all transform hover:scale-105 flex items-center space-x-2"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span className="hidden sm:inline">History</span>
                    </button>
                )}
            </div>
            
            <div className="flex-1 text-center">
                <div className="inline-block">
                    <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 mb-2 animate-gradient">
                        Political Sentiment AI
                    </h1>
                    <div className="h-1 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 rounded-full mx-auto mb-3"></div>
                </div>
                <p className="text-base md:text-lg text-gray-400">Advanced AI-powered political analysis and sentiment tracking</p>
            </div>
            
            <div>
                {token ? (
                    <button 
                        onClick={onLogout} 
                        className="bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white font-semibold py-2.5 px-5 rounded-xl transition-all transform hover:scale-105 flex items-center space-x-2 shadow-lg"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                        </svg>
                        <span className="hidden sm:inline">Logout</span>
                    </button>
                ) : (
                    <button 
                        onClick={onShowAuth} 
                        className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-2.5 px-5 rounded-xl transition-all transform hover:scale-105 flex items-center space-x-2 shadow-lg"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
                        </svg>
                        <span className="hidden sm:inline">Login</span>
                    </button>
                )}
            </div>
        </nav>
    </header>
);

// =================================================================================
// --- MAIN APP COMPONENT (ROUTER) ---
// =================================================================================
function App() {
    const [token, setToken] = useState(localStorage.getItem('authToken'));
    const [page, setPage] = useState(window.location.pathname);
    const [showAuth, setShowAuth] = useState(false);
    const [showHistory, setShowHistory] = useState(false);
    const [history, setHistory] = useState([]);

    const fetchHistory = async () => {
        if (!localStorage.getItem('authToken')) return;
        try {
            const response = await api.get('/history');
            setHistory(response.data);
        } catch (error) { console.error("Failed to fetch history:", error); }
    };
    
    useEffect(() => {
        const handleShowAuth = () => setShowAuth(true);
        const handleRefreshHistory = () => fetchHistory();

        window.addEventListener('show-auth', handleShowAuth);
        window.addEventListener('refresh-history', handleRefreshHistory);
        
        if (token) fetchHistory();
        
        return () => {
            window.removeEventListener('show-auth', handleShowAuth);
            window.removeEventListener('refresh-history', handleRefreshHistory);
        };
    }, [token]);

    const handleLoginSuccess = (newToken) => {
        localStorage.setItem('authToken', newToken);
        setToken(newToken);
        setShowAuth(false);
        fetchHistory();
    };

    const handleLogout = () => {
        setToken(null);
        localStorage.removeItem('authToken');
        localStorage.removeItem('jobId');
        setHistory([]);
        window.location.href = '/';
    };
    
    const handleDeleteHistoryItem = async (idToDelete) => {
        if (!window.confirm("Are you sure you want to delete this analysis?")) return;
        try {
            await api.delete(`/history/single/${idToDelete}`);
            setHistory(prev => prev.filter(item => item.job_id !== idToDelete));
        } catch (error) { alert("Could not delete item."); }
    };

    const handleClearAllHistory = async () => {
        if (!window.confirm("Delete ALL history? This action cannot be undone.")) return;
        try {
            await api.delete('/history/all');
            setHistory([]);
        } catch (error) { alert("Could not clear history."); }
    };

    const renderPage = () => {
        switch (page) {
            case '/verify-email': return <VerificationPage />;
            case '/forgot-password': return <ForgotPasswordPage />;
            case '/reset-password': return <ResetPasswordPage />;
            case '/': return <AnalysisPage token={token} />;
            default: return <NotFoundPage />;
        }
    };

    return (
        <div className="bg-gradient-to-br from-gray-900 via-gray-900 to-gray-800 text-white min-h-screen font-sans">
            <style>{`
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                @keyframes slideIn {
                    from { transform: translateY(20px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }
                @keyframes slideInLeft {
                    from { transform: translateX(-100%); }
                    to { transform: translateX(0); }
                }
                @keyframes scaleIn {
                    from { transform: scale(0.9); opacity: 0; }
                    to { transform: scale(1); opacity: 1; }
                }
                @keyframes shake {
                    0%, 100% { transform: translateX(0); }
                    25% { transform: translateX(-10px); }
                    75% { transform: translateX(10px); }
                }
                @keyframes gradient {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }
                .animate-fadeIn {
                    animation: fadeIn 0.5s ease-out;
                }
                .animate-slideIn {
                    animation: slideIn 0.3s ease-out;
                }
                .animate-slideInLeft {
                    animation: slideInLeft 0.3s ease-out;
                }
                .animate-scaleIn {
                    animation: scaleIn 0.3s ease-out;
                }
                .animate-shake {
                    animation: shake 0.5s ease-out;
                }
                .animate-gradient {
                    background-size: 200% 200%;
                    animation: gradient 3s ease infinite;
                }
                .line-clamp-4 {
                    display: -webkit-box;
                    -webkit-line-clamp: 4;
                    -webkit-box-orient: vertical;
                    overflow: hidden;
                }
            `}</style>
            
            {showAuth && <AuthModal onLoginSuccess={handleLoginSuccess} onClose={() => setShowAuth(false)} />}
            {showHistory && (
                <>
                    <div 
                        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 animate-fadeIn" 
                        onClick={() => setShowHistory(false)}
                    ></div>
                    <HistoryPanel 
                        history={history} 
                        onClose={() => setShowHistory(false)} 
                        onClearAll={handleClearAllHistory} 
                        onDeleteItem={handleDeleteHistoryItem} 
                    />
                </>
            )}
            
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <Header 
                    token={token} 
                    onLogout={handleLogout} 
                    onShowAuth={() => setShowAuth(true)} 
                    onShowHistory={() => setShowHistory(true)} 
                />
                {renderPage()}
            </div>
            
            <footer className="text-center py-8 text-gray-500 text-sm border-t border-gray-800 mt-20">
                <p>&copy; 2025 Political Sentiment AI. All rights reserved.</p>
                <p className="mt-2">Powered by Advanced Machine Learning</p>
            </footer>
        </div>
    );
}

export default App;