// Utilitários
const utils = {
    // Formata números para exibição
    formatNumber: (num) => {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    },

    // Formata duração para exibição (PT)
    formatDuration: (duration) => {
        const match = duration.match(/PT(\d+H)?(\d+M)?(\d+S)?/);
        if (!match) return '0:00';

        const hours = (match[1] || '').replace('H', '') || '0';
        const minutes = (match[2] || '').replace('M', '') || '0';
        const seconds = (match[3] || '').replace('S', '') || '0';
        
        if (hours !== '0') {
            return `${hours}:${minutes.padStart(2, '0')}:${seconds.padStart(2, '0')}`;
        }
        return `${minutes}:${seconds.padStart(2, '0')}`;
    },

    // Mostra notificação
    showNotification: (message, type = 'info') => {
        const notification = document.createElement('div');
        notification.className = `notification ${type} animate-slide-in`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        setTimeout(() => notification.classList.add('show'), 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
};

// Gerenciador de Filtros
class FilterManager {
    constructor() {
        this.filters = {};
        
        // Inicializa filtros dinamicamente a partir dos checkboxes existentes no DOM
        document.querySelectorAll('.filter-enabled').forEach(checkbox => {
            const filterName = checkbox.id.replace('filter-enabled-', '');
            // Define todos os filtros como habilitados por padrão
            checkbox.checked = true;
            this.filters[filterName] = {
                enabled: true,
                value: this.getInitialFilterValue(filterName)
            };
        });

        // Configura valores específicos para cada filtro
        this.setFilterDefaultValues();

        this.initializeEventListeners();
    }
    
    setFilterDefaultValues() {
        // Define valores padrão específicos para cada filtro
        const defaultValues = {
            'Tópicos Educacionais': 100,  // Máximo (conforme solicitado)
            'Toxicidade': 0,              // Mínimo (conforme solicitado)
            'Linguagem Imprópria': 0,     // Mínimo (conforme solicitado)
            'Diversidade': 100,
            'Interatividade': 100,
            'Engajamento': 100,
            'Sentimento': 100,
            'Conteúdo Sensível': 0
        };
        
        // Atualiza os valores no objeto filters
        for (const [filterName, value] of Object.entries(defaultValues)) {
            if (this.filters[filterName]) {
                this.filters[filterName].value = value;
                
                // Atualiza também os elementos visuais
                const rangeInput = document.querySelector(`.filter-range[data-filter="${filterName}"]`);
                if (rangeInput) {
                    rangeInput.value = value;
                }
            }
        }
    }

    getInitialFilterValue(filterName) {
        // Verifica se existe um radio button para este filtro
        const durationRadio = document.querySelector('input[name="duration"]:checked');
        const ageRadio = document.querySelector('input[name="age"]:checked');
        
        if (filterName === 'Duração' && durationRadio) {
            return durationRadio.value;
        } else if (filterName === 'Faixa Etária' && ageRadio) {
            return ageRadio.value;
        }
        
        // Para sliders, obtém o valor do elemento range
        const rangeInput = document.querySelector(`.filter-range[data-filter="${filterName}"]`);
        if (rangeInput) {
            return parseInt(rangeInput.value);
        }
        
        // Valor padrão
        return 50;
    }

    initializeEventListeners() {
        // Checkboxes para habilitar/desabilitar filtros
        document.querySelectorAll('.filter-enabled').forEach(checkbox => {
            const filterName = checkbox.id.replace('filter-enabled-', '');
            
            // Verifica se o filtro existe antes de acessar suas propriedades
            if (this.filters[filterName]) {
                checkbox.checked = this.filters[filterName].enabled;
                
                checkbox.addEventListener('change', (e) => {
                    this.filters[filterName].enabled = e.target.checked;
                });
            }
        });

        // Radio buttons para categorias
        document.querySelectorAll('input[type="radio"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                const filterName = this.getFilterNameFromRadio(e.target);
                if (filterName && this.filters[filterName]) {
                    this.filters[filterName].value = e.target.value;
                }
            });
        });

        // Range sliders
        document.querySelectorAll('.filter-range').forEach(range => {
            const filterName = range.dataset.filter;
            
            // Verifica se o filtro existe antes de acessar suas propriedades
            if (this.filters[filterName]) {
                range.value = this.filters[filterName].value;
                
                range.addEventListener('input', (e) => {
                    this.filters[filterName].value = parseInt(e.target.value);
                });
            }
        });

        // Botões para marcar/desmarcar todos
        document.getElementById('enable-all-filters').addEventListener('click', () => {
            this.toggleAllFilters(true);
        });

        document.getElementById('disable-all-filters').addEventListener('click', () => {
            this.toggleAllFilters(false);
        });
    }

    getFilterNameFromRadio(radio) {
        if (radio.name === 'duration') return 'Duração';
        if (radio.name === 'age') return 'Faixa Etária';
        return null;
    }

    toggleAllFilters(enabled) {
        console.log(`Alterando estado de todos os filtros para: ${enabled ? 'habilitado' : 'desabilitado'}`);
        
        // Primeiro atualizamos o estado nos checkboxes visuais
        document.querySelectorAll('.filter-enabled').forEach(checkbox => {
            checkbox.checked = enabled;
            const filterName = checkbox.id.replace('filter-enabled-', '');
            
            // Em seguida, atualizamos o estado no objeto de filtros
            if (this.filters[filterName]) {
                this.filters[filterName].enabled = enabled;
                console.log(`Filtro ${filterName} ${enabled ? 'habilitado' : 'desabilitado'}`);
            }
        });
    }

    getFilterWeights() {
        const weights = {};
        console.log("Coletando filtros habilitados:");
        
        for (const [name, filter] of Object.entries(this.filters)) {
            console.log(`- Filtro ${name}: ${filter.enabled ? 'habilitado' : 'desabilitado'}`);
            
            // Só inclui filtros que estão marcados como habilitados
            if (!filter.enabled) {
                console.log(`  - Filtro ${name} ignorado (desabilitado)`);
                continue;
            }

            // Determina o valor do filtro baseado no tipo
            if (typeof filter.value === 'number') {
                weights[name] = filter.value / 100;
                console.log(`  - Adicionado ${name} com valor numérico: ${filter.value / 100}`);
            } else {
                weights[name] = {
                    type: filter.value,
                    weight: 1
                };
                console.log(`  - Adicionado ${name} com tipo: ${filter.value}`);
            }
        }
        
        // Log do resultado final
        console.log("Filtros habilitados para envio:", weights);
        
        return weights;
    }
}

// Gerenciador de Busca
class SearchManager {
    constructor() {
        this.filterManager = new FilterManager();
        this.searchInput = document.getElementById('search-input');
        this.searchButton = document.getElementById('search-button');
        this.resultsSection = document.getElementById('results-section');
        this.videoGrid = document.getElementById('video-grid');
        this.videoTemplate = document.getElementById('video-card-template');
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.loadingSpinner = document.querySelector('.loading-spinner');
        this.isSearching = false;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Remove o evento de input que causava busca automática
        this.searchButton.addEventListener('click', () => {
            if (this.searchInput.value.trim()) {
                this.performSearch();
            }
        });

        // Prevenir envio do formulário
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (this.searchInput.value.trim()) {
                    this.performSearch();
                }
            }
        });
    }

    showLoading() {
        this.loadingOverlay.classList.remove('hidden');
        this.loadingOverlay.classList.add('flex');
        this.loadingOverlay.classList.add('show');
        this.loadingSpinner.classList.remove('hidden');
        this.searchButton.disabled = true;
    }

    hideLoading() {
        this.loadingOverlay.classList.remove('show');
        setTimeout(() => {
            this.loadingOverlay.classList.add('hidden');
            this.loadingOverlay.classList.remove('flex');
        }, 300);
        this.loadingSpinner.classList.add('hidden');
        this.searchButton.disabled = false;
    }

    async performSearch() {
        const query = this.searchInput.value.trim();
        if (!query) {
            utils.showNotification('Por favor, digite um termo de busca', 'warning');
            return;
        }
        
        // Evita múltiplas requisições simultâneas
        if (this.isSearching) {
            return;
        }
        
        this.isSearching = true;
        this.showLoading();
        
        try {
            // Obtém os filtros habilitados
            const filterWeights = this.filterManager.getFilterWeights();
            console.log('\n=== Search Request Details ===');
            console.log('Query:', query);
            console.log('Enabled filters:', filterWeights);
            
            const params = new URLSearchParams({
                query: query,
                filter_weights: JSON.stringify(filterWeights)
            });
            
            const apiUrl = `/api/v1/videos/search/?${params}`;
            console.log('Request URL:', apiUrl);
            
            const response = await fetch(apiUrl);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('API Response:', data);
            
            this.displayResults(data.videos);
            
        } catch (error) {
            console.error('Search error:', error);
            utils.showNotification('Erro ao buscar vídeos', 'error');
            
        } finally {
            this.hideLoading();
            this.isSearching = false;
        }
    }

    displayResults(videos) {
        this.videoGrid.innerHTML = '';
        this.resultsSection.classList.remove('hidden');
        
        if (!videos || videos.length === 0) {
            const noResults = document.createElement('div');
            noResults.className = 'col-span-full text-center text-gray-600 py-8';
            noResults.innerHTML = `
                <p class="text-xl">Nenhum vídeo encontrado</p>
                <p class="mt-2">Tente uma busca diferente ou ajuste os filtros</p>
            `;
            this.videoGrid.appendChild(noResults);
            return;
        }
        
        // Mostra os vídeos encontrados
        console.log(`\n=== Exibindo ${videos.length} vídeos nos resultados ===`);
        
        videos.forEach((video, index) => {
            try {
                // Verificar se o vídeo tem as propriedades necessárias
                if (!video || !video.id || !video.title) {
                    console.warn('Vídeo inválido nos resultados:', video);
                    return;
                }
                
                // Log detalhado do vídeo e seus scores
                console.log(`\nVídeo ${index + 1}: ${video.title}`);
                console.log(`ID: ${video.id}`);
                console.log(`Duração: ${video.duration || 'N/A'} (${video.duration_seconds || 0} segundos)`);
                
                if (video.filter_scores) {
                    console.log('Scores dos filtros:');
                    Object.entries(video.filter_scores).forEach(([filter, score]) => {
                        console.log(`- ${filter}: ${(score * 100).toFixed(1)}%`);
                    });
                    console.log(`Score final: ${(video.final_score * 100).toFixed(1)}%`);
                } else {
                    console.log('Nenhum score de filtro disponível');
                }
                
                // Clone o template
                const template = this.videoTemplate.content.cloneNode(true);
                
                // Preenche os dados do vídeo
                const card = template.querySelector('.video-card');
                const title = template.querySelector('.video-title');
                const thumbnail = template.querySelector('.video-thumbnail');
                const channelName = template.querySelector('.video-channel');
                const viewCount = template.querySelector('.video-views');
                const duration = template.querySelector('.video-duration');
                const score = template.querySelector('.video-score');
                
                // ID para link
                card.dataset.videoId = video.id;
                card.href = `https://www.youtube.com/watch?v=${video.id}`;
                
                // Título
                title.textContent = video.title;
                
                // Thumbnail
                if (video.thumbnail) {
                    thumbnail.src = video.thumbnail;
                    thumbnail.alt = video.title;
                } else {
                    thumbnail.src = '/static/img/placeholder.jpg';
                }
                
                // Canal
                if (video.channel_title) {
                    channelName.textContent = video.channel_title;
                } else {
                    channelName.textContent = 'Canal desconhecido';
                }
                
                // Views
                if (video.view_count !== undefined) {
                    viewCount.textContent = utils.formatNumber(video.view_count) + ' visualizações';
                } else {
                    viewCount.textContent = 'Views indisponíveis';
                }
                
                // Duração
                if (video.duration) {
                    duration.textContent = utils.formatDuration(video.duration);
                } else if (video.duration_seconds) {
                    // Converter segundos para formato mm:ss
                    const minutes = Math.floor(video.duration_seconds / 60);
                    const seconds = Math.floor(video.duration_seconds % 60);
                    duration.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                } else {
                    duration.textContent = '--:--';
                }
                
                // Score
                if (video.final_score !== undefined) {
                    const scoreValue = Math.round(video.final_score * 100);
                    score.textContent = `Score: ${scoreValue}%`;
                    
                    // Cor baseada no score
                    if (scoreValue >= 80) {
                        score.classList.add('bg-green-500');
                    } else if (scoreValue >= 60) {
                        score.classList.add('bg-yellow-500');
                    } else {
                        score.classList.add('bg-red-500');
                    }
                } else {
                    score.textContent = 'Score: N/A';
                    score.classList.add('bg-gray-500');
                }
                
                // Adiciona o card à grade
                this.videoGrid.appendChild(template);
                
            } catch (error) {
                console.error('Erro ao exibir vídeo:', error);
            }
        });
    }
}

// Inicialização
document.addEventListener('DOMContentLoaded', () => {
    window.searchManager = new SearchManager();
});