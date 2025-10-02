import { useEffect, useState } from 'react';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function VerificationPage() {
    const [message, setMessage] = useState('Verifying your email, please wait...');
    
    useEffect(() => {
        const verify = async () => {
            const token = new URLSearchParams(window.location.search).get('token');
            if (!token) {
                setMessage('Error: No verification token found.');
                return;
            }
            try {
                const response = await axios.post(`${API_BASE_URL}/verify-email?token=${token}`);
                setMessage(response.data.message + " Redirecting to login...");
                setTimeout(() => window.location.href = '/', 2000);
            } catch (error) {
                setMessage(error.response?.data?.detail || 'Verification failed.');
            }
        };
        verify();
    }, []);

    return (
        <div className="bg-gray-900 text-white min-h-screen flex items-center justify-center">
            <div className="bg-gray-800 p-8 rounded-lg shadow-xl text-center">
                <h2 className="text-2xl font-bold mb-4">Account Verification</h2>
                <p className="text-gray-400">{message}</p>
            </div>
        </div>
    );
}

export default VerificationPage;