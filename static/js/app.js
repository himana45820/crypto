// Global state
const app = {
    charts: {
        price: null,
        volatility: null
    },
    colors: {
        bitcoin: { border: '#f7931a', bg: 'rgba(247,147,26,0.15)' },
        ethereum: { border: '#627eea', bg: 'rgba(98,126,234,0.15)' },
        xrp: { border: '#ffffff', bg: 'rgba(255,255,255,0.15)' },
        default: { border: '#3E63FF', bg: 'rgba(62,99,255,0.15)' }
    },
    
    init: function() {
        // Set up event listeners for navigation
        document.querySelectorAll('.nav-links li').forEach(li => {
            li.addEventListener('click', (e) => {
                const page = e.currentTarget.getAttribute('data-page');
                this.switchPage(page);
                
                // Active state styling
                document.querySelectorAll('.nav-links li').forEach(el => el.classList.remove('active'));
                e.currentTarget.classList.add('active');
            });
        });

        // Set up prediction form
        document.getElementById('predict-form').addEventListener('submit', this.handlePrediction.bind(this));
    },

    switchPage: function(page) {
        // Hide all pages
        document.querySelectorAll('.page-section').forEach(el => {
            el.style.display = 'none';
        });

        // Update nav active state if switched from dashboard cards
        document.querySelectorAll('.nav-links li').forEach(el => {
            if(el.getAttribute('data-page') === page) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        });

        switch(page) {
            case 'home':
                document.getElementById('page-title').innerText = 'Dashboard Overview';
                document.getElementById('page-home').style.display = 'block';
                break;
            case 'predict':
                document.getElementById('page-title').innerText = 'Make a Prediction';
                document.getElementById('page-predict').style.display = 'block';
                break;
            case 'bitcoin':
            case 'ethereum':
            case 'xrp':
                document.getElementById('page-title').innerText = `${page.charAt(0).toUpperCase() + page.slice(1)} Analytics`;
                document.getElementById('chart-title').innerText = page.charAt(0).toUpperCase() + page.slice(1);
                document.getElementById('page-coin').style.display = 'block';
                this.loadCoinData(page);
                break;
        }
    },

    loadCoinData: async function(coin) {
        // Show loading
        document.getElementById('loading-overlay').style.display = 'block';
        document.querySelector('.metrics-row').style.display = 'none';
        document.querySelector('.charts-container').style.display = 'none';
        
        try {
            const res = await fetch(`/api/data/${coin}`);
            if(!res.ok) throw new Error("Failed to fetch data");
            const data = await res.json();
            
            // Render UI
            this.renderCoinMetrics(data);
            this.renderCharts(coin, data);
            
            document.getElementById('loading-overlay').style.display = 'none';
            document.querySelector('.metrics-row').style.display = 'grid';
            document.querySelector('.charts-container').style.display = 'flex';
        } catch (err) {
            console.error(err);
            document.getElementById('loading-overlay').innerHTML = `<p style="color:#ff6b6b">Error loading data. Is the Flask server running?</p>`;
        }
    },

    renderCoinMetrics: function(data) {
        const latestPrice = data.prices[data.prices.length - 1];
        const latestVol = data.volatility[data.volatility.length - 1];
        
        document.getElementById('coin-latest-price').innerText = `$${latestPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        document.getElementById('coin-latest-vol').innerText = `${(latestVol * 100).toFixed(2)}%`;
    },

    renderCharts: function(coin, data) {
        Chart.defaults.color = '#c5c6c7';
        Chart.defaults.font.family = "'Outfit', sans-serif";
        
        const colors = this.colors[coin] || this.colors.default;

        // Destroy old charts to recreate
        if (this.charts.price) this.charts.price.destroy();
        if (this.charts.volatility) this.charts.volatility.destroy();

        // Downsample data if too large for better performance rendering (take every Nth point)
        // For line charts with thousands of points
        const step = Math.max(1, Math.floor(data.dates.length / 500));
        const chartDates = data.dates.filter((_, i) => i % step === 0);
        const chartPrices = data.prices.filter((_, i) => i % step === 0);
        const chartVols = data.volatility.filter((_, i) => i % step === 0);

        // Price Chart
        const ctxPrice = document.getElementById('priceChart').getContext('2d');
        const gradientPrice = ctxPrice.createLinearGradient(0, 0, 0, 400);
        gradientPrice.addColorStop(0, colors.bg);
        gradientPrice.addColorStop(1, 'rgba(8,10,26,0)');

        this.charts.price = new Chart(ctxPrice, {
            type: 'line',
            data: {
                labels: chartDates,
                datasets: [{
                    label: 'Price (USD)',
                    data: chartPrices,
                    borderColor: colors.border,
                    backgroundColor: gradientPrice,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    fill: true,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });

        // Volatility Chart
        const ctxVol = document.getElementById('volatilityChart').getContext('2d');
        const gradientVol = ctxVol.createLinearGradient(0, 0, 0, 400);
        gradientVol.addColorStop(0, 'rgba(157, 78, 221, 0.2)');
        gradientVol.addColorStop(1, 'rgba(8,10,26,0)');

        this.charts.volatility = new Chart(ctxVol, {
            type: 'line',
            data: {
                labels: chartDates,
                datasets: [{
                    label: 'Asset Volatility',
                    data: chartVols,
                    borderColor: '#9D4EDD',
                    backgroundColor: gradientVol,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    fill: true,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Volatility: ${(context.raw * 100).toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    },

    handlePrediction: async function(e) {
        e.preventDefault();
        
        const coin = document.getElementById('predict-coin').value;
        const date = document.getElementById('predict-date').value;
        
        const btn = document.getElementById('predict-btn');
        const spinner = btn.querySelector('.btn-spinner');
        const btnText = btn.querySelector('span');
        const resultSection = document.getElementById('predict-result');
        const errorSection = document.getElementById('predict-error');
        
        // Reset states
        errorSection.style.display = 'none';
        resultSection.style.display = 'none';
        spinner.style.display = 'block';
        btnText.innerText = 'Calculating...';
        btn.disabled = true;
        
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ coin, date })
            });
            
            const data = await res.json();
            
            if(!res.ok) {
                throw new Error(data.error || 'Something went wrong');
            }
            
            // Display Results
            document.getElementById('res-coin').innerText = data.coin.toUpperCase();
            document.getElementById('res-date').innerText = new Date(data.target_date).toLocaleDateString();
            document.getElementById('res-price').innerText = `$${data.predicted_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            
            document.getElementById('res-last-price').innerText = `$${data.last_price.toLocaleString()}`;
            document.getElementById('res-last-date').innerText = new Date(data.last_date).toLocaleDateString();
            
            resultSection.style.display = 'block';
            
        } catch (err) {
            errorSection.innerText = err.message;
            errorSection.style.display = 'block';
        } finally {
            spinner.style.display = 'none';
            btnText.innerText = 'Generate Prediction';
            btn.disabled = false;
        }
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => app.init());
