define(['core/log'], function(Log) {
    const MAX_HISTORY = 80;
    const MAX_USER_MESSAGES = 80;

    const safeReadHistory = (key) => {
        try {
            const raw = window.localStorage.getItem(key);
            if (!raw) {
                return [];
            }
            const parsed = JSON.parse(raw);
            return Array.isArray(parsed) ? parsed : [];
        } catch (e) {
            return [];
        }
    };

    const safeWriteHistory = (key, value) => {
        try {
            window.localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            // Ignore storage failures.
        }
    };

    const normalizeMessageType = (type) => {
        const next = String(type || '').trim().toLowerCase();
        if (next === 'user') {
            return 'user';
        }
        return 'assistant';
    };

    const normalizeHistoryEntry = (entry) => {
        const source = (entry && typeof entry === 'object') ? entry : {};
        return {
            type: normalizeMessageType(source.type),
            text: String(source.text || ''),
            sources: Array.isArray(source.sources) ? source.sources.map((item) => String(item || '')) : [],
            requestId: String(source.requestId || ''),
            chatMode: String(source.chatMode || ''),
            questionId: String(source.questionId || ''),
            runId: Math.max(0, parseInt(source.runId, 10) || 0),
            questionText: String(source.questionText || ''),
            courseId: Math.max(0, parseInt(source.courseId, 10) || 0),
            topic: String(source.topic || '')
        };
    };

    const userMessageCount = (history) => history.filter((entry) => entry.type === 'user').length;

    const trimHistory = (history) => {
        const next = Array.isArray(history) ? history.map(normalizeHistoryEntry) : [];
        while (userMessageCount(next) > MAX_USER_MESSAGES && next.length > 0) {
            next.shift();
        }
        return next.slice(-MAX_HISTORY);
    };

    const buildConversationContext = (items, limit = 6) => {
        const list = Array.isArray(items) ? items.slice(-limit) : [];
        return list
            .map((entry) => {
                const type = normalizeMessageType(entry && entry.type);
                const text = String((entry && entry.text) || '').trim();
                if (!text) {
                    return null;
                }
                if (type === 'user') {
                    return {role: 'user', text: text};
                }
                return {role: 'assistant', text: text};
            })
            .filter(Boolean);
    };

    const postForm = async (url, data) => {
        const response = await fetch(url, {
            method: 'POST',
            body: data,
            credentials: 'same-origin',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        });
        const raw = await response.text();
        if (!raw || !raw.trim()) {
            throw new Error(`Empty response from server (HTTP ${response.status}).`);
        }
        try {
            return JSON.parse(raw);
        } catch (err) {
            const snippet = raw.slice(0, 180).replace(/\s+/g, ' ').trim();
            throw new Error(`Invalid JSON response (HTTP ${response.status}): ${snippet || 'no payload'}`);
        }
    };

    const appendMessageDom = (entry, config) => {
        const normalizedEntry = (entry && typeof entry === 'object') ? entry : {};
        normalizedEntry.type = normalizeMessageType(normalizedEntry.type);
        normalizedEntry.text = String(normalizedEntry.text || '');
        normalizedEntry.sources = Array.isArray(normalizedEntry.sources)
            ? normalizedEntry.sources.map((item) => String(item || ''))
            : [];
        const messages = document.getElementById('local-chatbot-messages');
        if (!messages) {
            return null;
        }

        const item = document.createElement('div');
        item.className = `local-chatbot-message ${normalizedEntry.type}`;

        const body = document.createElement('div');
        body.className = 'local-chatbot-message-body';
        body.textContent = normalizedEntry.text;
        item.appendChild(body);
        messages.appendChild(item);

        if (Array.isArray(normalizedEntry.sources) && normalizedEntry.sources.length > 0) {
            const source = document.createElement('div');
            source.className = 'local-chatbot-source';
            source.textContent = `source: ${normalizedEntry.sources.join(', ')}`;
            item.appendChild(source);
        }

        messages.scrollTop = messages.scrollHeight;
        return item;
    };

    const renderFiles = (files, nofiles, onFileClick, selectedFile) => {
        const wrap = document.getElementById('local-chatbot-files');
        if (!wrap) {
            return;
        }

        if (!Array.isArray(files) || files.length === 0) {
            wrap.innerHTML = `<p class="local-chatbot-empty">${nofiles}</p>`;
            return;
        }

        wrap.innerHTML = '';
        files.forEach((file) => {
            const name = String((file && file.name) || '');
            if (name === '') {
                return;
            }
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'local-chatbot-file-item';
            item.dataset.file = name;
            if (selectedFile === name) {
                item.classList.add('active');
            }
            item.innerHTML = `<span>${name}</span>`;
            item.addEventListener('click', () => onFileClick(name));
            wrap.appendChild(item);
        });
    };

    return {
        init: function(config) {
            Log.debug('local_chatbot page initialized');

            const input = document.getElementById('local-chatbot-input');
            const sendBtn = document.getElementById('local-chatbot-send');
            const clearBtn = document.getElementById('local-chatbot-clear');
            const usageWrap = document.getElementById('local-chatbot-usage');
            const statusWrap = document.getElementById('local-chatbot-status');
            const previewBody = document.getElementById('local-chatbot-preview-body');
            const previewName = document.getElementById('local-chatbot-preview-name');
            const refreshEmbeddingBtn = document.getElementById('local-chatbot-refresh-embedding-btn');
            const previewEmbeddingStatus = document.getElementById('local-chatbot-preview-embedding-status');
            const chatClassInput = document.getElementById('local-chatbot-chat-class');
            const chatTopicInput = document.getElementById('local-chatbot-chat-topic');
            const uploadInput = document.getElementById('local-chatbot-upload-input');
            const uploadBtn = document.getElementById('local-chatbot-upload-btn');
            const clearUploadBtn = document.getElementById('local-chatbot-clear-upload-btn');
            const materialContextWrap = document.getElementById('local-chatbot-material-context');
            const embeddingConfigWrap = document.getElementById('local-chatbot-embedding-config');
            const modeInputs = Array.from(document.querySelectorAll('[data-mode-value]'));
            const evalModeInput = document.getElementById('local-chatbot-eval-mode');
            const evalControlsWrap = document.getElementById('local-chatbot-eval-controls');
            const evalSourceInputs = Array.from(document.querySelectorAll('input[name="local-chatbot-eval-source"]'));
            const questionIdInput = document.getElementById('local-chatbot-question-id');
            const runIdInput = document.getElementById('local-chatbot-run-id');
            const evalDatasetFileInput = document.getElementById('local-chatbot-eval-dataset-file');
            const evalDatasetRunsInput = document.getElementById('local-chatbot-eval-dataset-runs');
            const evalDatasetRunBtn = document.getElementById('local-chatbot-eval-dataset-run');
            const appRoot = document.getElementById(String(config.approotid || 'local-chatbot-app').trim());
            const appOwnsChat = Boolean(config.appownschat && appRoot);
            const appOwnsMaterialsPreview = Boolean(config.appownsmaterialspreview && appRoot);

            const storageKey = `local_chatbot_history_u${config.userid || 'anon'}`;
            const configuredCourseTopics = (config && typeof config.coursetopics === 'object' && config.coursetopics !== null)
                ? config.coursetopics
                : {};

            let history = trimHistory(safeReadHistory(storageKey));
            let selectedFile = null;
            let activeFiles = [];
            let materialContext = {
                mode: 'none',
                is_manual: false,
                disable_topic_select: false
            };
            let parseStatus = {status: 'no_materials', is_parsed: false, parsed_at: 0, sources: 0};
            let selectedEmbeddingStatus = null;
            let isChatBusy = false;
            let isDatasetBusy = false;
            let isRefreshEmbeddingBusy = false;

            const getAppChatActions = () => {
                if (!appOwnsChat || !appRoot || !appRoot.__localChatbotApp || !appRoot.__localChatbotApp.actions) {
                    return null;
                }
                return appRoot.__localChatbotApp.actions;
            };

            const appendOwnedChatEntry = (entry) => {
                const actions = getAppChatActions();
                const normalizedEntry = normalizeHistoryEntry(entry);
                history = trimHistory(history.concat([normalizedEntry]));
                if (actions && typeof actions.appendHistoryEntry === 'function') {
                    actions.appendHistoryEntry(normalizedEntry);
                    return;
                }
                appendMessageDom(normalizedEntry, config, {});
            };

            const syncMaterialsFromAppDetail = (detail = {}) => {
                const nextContext = (detail.materialContext && typeof detail.materialContext === 'object')
                    ? detail.materialContext
                    : {};
                materialContext = {
                    mode: String(nextContext.mode || 'none').trim().toLowerCase() || 'none',
                    is_manual: Boolean(nextContext.is_manual),
                    disable_topic_select: Boolean(nextContext.disable_topic_select),
                    course_id: Number(nextContext.course_id || 0),
                    topic: String(nextContext.topic || '').trim()
                };
                activeFiles = Array.isArray(detail.activeFiles) ? detail.activeFiles : [];
                parseStatus = (detail.parseStatus && typeof detail.parseStatus === 'object')
                    ? detail.parseStatus
                    : {status: 'no_materials', is_parsed: false, parsed_at: 0, sources: 0};
                selectedFile = String(detail.selectedFile || '').trim();
                selectedEmbeddingStatus = detail.selectedEmbeddingStatus || null;
            };

            const syncMaterialsFromAppState = (appState = {}) => {
                const nextState = (appState && typeof appState === 'object') ? appState : {};
                const nextContext = (nextState.materialContext && typeof nextState.materialContext === 'object')
                    ? nextState.materialContext
                    : {};
                syncMaterialsFromAppDetail({
                    materialContext: {
                        mode: nextContext.mode,
                        is_manual: nextContext.isManual,
                        disable_topic_select: nextContext.disableTopicSelect,
                        course_id: nextContext.courseId,
                        topic: nextContext.topic
                    },
                    activeFiles: nextState.materialsState && Array.isArray(nextState.materialsState.activeFiles)
                        ? nextState.materialsState.activeFiles
                        : [],
                    selectedFile: nextState.materialsState ? nextState.materialsState.selectedFile : '',
                    parseStatus: nextState.materialsState ? nextState.materialsState.parseStatus : null,
                    selectedEmbeddingStatus: nextState.materialsState
                        ? nextState.materialsState.selectedEmbeddingStatus
                        : null
                });
            };

            const getSelectedModes = () => modeInputs
                .filter((entry) => entry && entry.checked)
                .map((entry) => String(entry.dataset.modeValue || '').trim())
                .filter(Boolean);

            const getEvaluationSource = () => {
                const active = evalSourceInputs.find((entry) => entry && entry.checked);
                return active ? String(active.value || 'chat').trim().toLowerCase() : 'chat';
            };

            const getPrimaryMode = () => {
                const selectedModes = getSelectedModes();
                if (selectedModes.length > 0) {
                    return selectedModes[0];
                }
                return String(config.defaultchatmode || 'rag_ollama');
            };

            const resolveActiveEmbeddingText = () => {
                const primaryMode = getPrimaryMode();
                if (primaryMode === 'llm_only') {
                    return String(config.embeddingconfigllmonly || 'No embedding is used in LLM-only mode.');
                }
                if (primaryMode === 'rag_ollama') {
                    return `Ollama: ${String(config.embedmodelollama || 'nomic-embed-text')}`;
                }
                if (primaryMode === 'rag_bert' || primaryMode === 'rag_msmarco') {
                    return `BERT: ${String(config.embedmodelbert || 'sentence-transformers/msmarco-bert-base-dot-v5')}`;
                }
                return `Auto: ${String(config.embedbackenddefault || 'auto')}`;
            };

            const renderEmbeddingConfig = () => {
                if (!embeddingConfigWrap) {
                    return;
                }
                embeddingConfigWrap.innerHTML =
                    `<div><strong>${String(config.embeddingconfigtitle || 'Embedding configuration')}</strong></div>`
                    + `<div><strong>${String(config.embeddingconfigactive || 'Active embedding')}:</strong> ${resolveActiveEmbeddingText()}</div>`
                    + `<div><strong>${String(config.embeddingconfigbackend || 'Default backend')}:</strong> ${String(config.embedbackenddefault || 'auto')}</div>`
                    + `<div><strong>${String(config.embeddingconfigollama || 'Ollama embedding model')}:</strong> ${String(config.embedmodelollama || 'nomic-embed-text')}</div>`
                    + `<div><strong>${String(config.embeddingconfigbert || 'BERT embedding model')}:</strong> ${String(config.embedmodelbert || 'sentence-transformers/msmarco-bert-base-dot-v5')}</div>`;
            };

            const getActivePreviewFilename = () => {
                const fromState = String(selectedFile || '').trim();
                if (fromState !== '') {
                    return fromState;
                }
                const fromPreview = previewName ? String(previewName.textContent || '').trim() : '';
                if (fromPreview !== '' && fromPreview !== '-') {
                    return fromPreview;
                }
                return '';
            };

            const formatUnixTime = (value) => {
                const numeric = Math.max(0, parseInt(value, 10) || 0);
                if (!numeric) {
                    return '-';
                }
                try {
                    return new Date(numeric * 1000).toLocaleString();
                } catch (error) {
                    return '-';
                }
            };

            const humanizeParseStatus = (value) => {
                const normalized = String(value || '').trim().toLowerCase();
                if (normalized === 'parsed') {
                    return 'parsed';
                }
                if (normalized === 'needs_parsing') {
                    return 'needs parsing';
                }
                if (normalized === 'no_materials') {
                    return 'no materials';
                }
                return normalized || '-';
            };

            const renderPreviewEmbeddingStatus = (info = null) => {
                if (!previewEmbeddingStatus) {
                    return;
                }
                const effectiveInfo = (info && typeof info === 'object') ? info : null;
                const activePreviewFilename = getActivePreviewFilename();
                if (!activePreviewFilename) {
                    previewEmbeddingStatus.innerHTML = 'Select a file to view embedding status.';
                    return;
                }

                const fileInActiveCorpus = effectiveInfo
                    ? Boolean(effectiveInfo.file_in_active_corpus)
                    : activeFiles.some((file) => String((file && file.name) || '') === activePreviewFilename);
                const isEmbedded = effectiveInfo
                    ? Boolean(effectiveInfo.is_embedded_in_active_index)
                    : (fileInActiveCorpus && Boolean(parseStatus.is_parsed));
                const statusValue = effectiveInfo
                    ? String(effectiveInfo.parse_status || '')
                    : String(parseStatus.status || '');
                const parsedAt = effectiveInfo
                    ? formatUnixTime(effectiveInfo.parsed_at)
                    : formatUnixTime(parseStatus.parsed_at);
                const embeddingModel = effectiveInfo && effectiveInfo.embedding_model
                    ? String(effectiveInfo.embedding_model)
                    : String(config.embedmodelbert || config.embedmodelollama || '-');

                previewEmbeddingStatus.innerHTML =
                    `<div><strong>Selected file:</strong> ${activePreviewFilename}</div>`
                    + `<div><strong>Index scope:</strong> active corpus</div>`
                    + `<div><strong>File in active corpus:</strong> ${fileInActiveCorpus ? 'yes' : 'no'}</div>`
                    + `<div><strong>Embedded in current index:</strong> ${isEmbedded ? 'yes' : 'no'}</div>`
                    + `<div><strong>Index status:</strong> ${humanizeParseStatus(statusValue)}</div>`
                    + `<div><strong>Current embedding model:</strong> ${embeddingModel}</div>`
                    + `<div><strong>Last indexed:</strong> ${parsedAt}</div>`;
            };

            const updateUsage = () => {
                if (!usageWrap) {
                    return;
                }
                usageWrap.textContent = `${config.chatusagelabel || 'Usage'}: ${userMessageCount(history)}/${MAX_USER_MESSAGES}`;
            };

            const persistHistory = () => {
                history = trimHistory(history);
                safeWriteHistory(storageKey, history);
                updateUsage();
            };

            const renderHistory = () => {
                const messages = document.getElementById('local-chatbot-messages');
                if (!messages) {
                    return;
                }
                messages.innerHTML = '';
                if (!history.length) {
                    appendMessageDom({
                        type: 'assistant',
                        text: config.defaultgreeting || '',
                        sources: []
                    }, config);
                    return;
                }
                history.forEach((entry) => {
                    appendMessageDom(entry, config);
                });
            };

            const resetPreview = (message) => {
                if (previewName) {
                    previewName.textContent = '-';
                }
                if (previewBody) {
                    previewBody.innerHTML = `<p class="local-chatbot-empty">${message}</p>`;
                }
            };

            const setStatus = (text) => {
                if (statusWrap) {
                    statusWrap.textContent = String(text || '');
                }
            };

            const applyMaterialContext = (context) => {
                const next = (context && typeof context === 'object') ? context : {};
                materialContext = {
                    mode: String(next.mode || 'none').trim().toLowerCase() || 'none',
                    is_manual: Boolean(next.is_manual),
                    disable_topic_select: Boolean(next.disable_topic_select),
                    course_id: Number(next.course_id || 0),
                    topic: String(next.topic || '').trim()
                };
                if (materialContext.is_manual) {
                    if (chatClassInput) {
                        chatClassInput.value = '';
                    }
                    if (chatTopicInput) {
                        chatTopicInput.innerHTML = '';
                        const placeholder = document.createElement('option');
                        placeholder.value = '';
                        placeholder.textContent = config.coursetopicplaceholder || 'Select topic';
                        chatTopicInput.appendChild(placeholder);
                        chatTopicInput.value = '';
                    }
                    return;
                }
                if (materialContext.mode === 'topic' && materialContext.course_id > 0) {
                    if (chatClassInput) {
                        chatClassInput.value = String(materialContext.course_id);
                    }
                    populateTopicOptions(String(materialContext.course_id));
                    if (chatTopicInput && materialContext.topic) {
                        chatTopicInput.value = materialContext.topic;
                    }
                }
            };

            const renderMaterialContextNotice = () => {
                if (appOwnsMaterialsPreview) {
                    return;
                }
                if (!materialContextWrap) {
                    return;
                }
                if (materialContext.is_manual) {
                    materialContextWrap.textContent = String(
                        config.manualmodeactive || 'Manual upload is active. Clear uploaded materials to use class/topic materials again.'
                    );
                    materialContextWrap.classList.remove('local-chatbot-hidden');
                    return;
                }
                materialContextWrap.textContent = '';
                materialContextWrap.classList.add('local-chatbot-hidden');
            };

            const syncMaterialControlsState = () => {
                if (appOwnsMaterialsPreview) {
                    return;
                }
                const disableTopicSelect = Boolean(materialContext.disable_topic_select || materialContext.is_manual);
                const activePreviewFilename = getActivePreviewFilename();
                if (chatClassInput) {
                    chatClassInput.disabled = disableTopicSelect;
                }
                if (chatTopicInput) {
                    chatTopicInput.disabled = disableTopicSelect;
                }
                if (clearUploadBtn) {
                    clearUploadBtn.disabled = !config.canmanualupload || !materialContext.is_manual;
                }
                if (refreshEmbeddingBtn) {
                    refreshEmbeddingBtn.disabled = !activePreviewFilename || isRefreshEmbeddingBusy;
                }
                renderMaterialContextNotice();
            };

            const syncEvalControlsVisibility = () => {
                if (!evalControlsWrap || !evalModeInput) {
                    return;
                }
                if (evalModeInput.checked) {
                    evalControlsWrap.classList.remove('local-chatbot-hidden');
                } else {
                    evalControlsWrap.classList.add('local-chatbot-hidden');
                }
            };

            const enforceModeSelectionRules = (changedInput = null) => {
                if (!modeInputs.length) {
                    renderEmbeddingConfig();
                    return;
                }
                const evaluationEnabled = Boolean(evalModeInput && evalModeInput.checked);
                const evaluationSource = getEvaluationSource();

                if (evaluationEnabled && evaluationSource === 'chat') {
                    if (changedInput && changedInput.checked) {
                        modeInputs.forEach((entry) => {
                            if (entry !== changedInput) {
                                entry.checked = false;
                            }
                        });
                    }
                    const selected = modeInputs.filter((entry) => entry.checked);
                    if (!selected.length) {
                        const fallbackMode = String(config.defaultchatmode || 'rag_ollama');
                        const fallback = modeInputs.find((entry) => String(entry.dataset.modeValue || '').trim() === fallbackMode) || modeInputs[0];
                        if (fallback) {
                            fallback.checked = true;
                        }
                    } else if (selected.length > 1) {
                        selected.slice(1).forEach((entry) => {
                            entry.checked = false;
                        });
                    }
                    renderEmbeddingConfig();
                    return;
                }

                if (!getSelectedModes().length) {
                    const fallbackMode = String(config.defaultchatmode || 'rag_ollama');
                    const fallback = modeInputs.find((entry) => String(entry.dataset.modeValue || '').trim() === fallbackMode) || modeInputs[0];
                    if (fallback) {
                        fallback.checked = true;
                    }
                }
                renderEmbeddingConfig();
            };

            const syncEvaluationSourceState = () => {
                const evaluationEnabled = Boolean(evalModeInput && evalModeInput.checked);
                const evaluationSource = getEvaluationSource();
                const useDataset = evaluationEnabled && evaluationSource === 'dataset';
                const useDirectChat = evaluationEnabled && evaluationSource === 'chat';

                evalSourceInputs.forEach((entry) => {
                    entry.disabled = !evaluationEnabled || isChatBusy || isDatasetBusy;
                });

                if (questionIdInput) {
                    questionIdInput.disabled = !useDirectChat || isChatBusy;
                }
                if (runIdInput) {
                    runIdInput.disabled = !useDirectChat || isChatBusy;
                }
                if (evalDatasetFileInput) {
                    evalDatasetFileInput.disabled = !useDataset || isDatasetBusy;
                }
                if (evalDatasetRunsInput) {
                    evalDatasetRunsInput.disabled = !useDataset || isDatasetBusy;
                }
                if (evalDatasetRunBtn) {
                    evalDatasetRunBtn.disabled = !useDataset || isDatasetBusy;
                }
                if (input) {
                    input.disabled = useDataset || isChatBusy;
                }
                if (sendBtn) {
                    sendBtn.disabled = useDataset || isChatBusy;
                }

                enforceModeSelectionRules();
            };

            const populateTopicOptions = (courseValue) => {
                if (!chatTopicInput) {
                    return;
                }
                const topics = configuredCourseTopics[String(courseValue || '')] || [];
                chatTopicInput.innerHTML = '';

                const placeholder = document.createElement('option');
                placeholder.value = '';
                placeholder.textContent = config.coursetopicplaceholder || 'Select topic';
                chatTopicInput.appendChild(placeholder);

                topics.forEach((topic) => {
                    const value = String((topic && topic.value) || (topic && topic.label) || '').trim();
                    if (value === '') {
                        return;
                    }
                    const option = document.createElement('option');
                    option.value = value;
                    option.textContent = String((topic && topic.label) || value);
                    chatTopicInput.appendChild(option);
                });
            };

            const refreshMaterialView = (files, context, incomingParseStatus = null) => {
                activeFiles = Array.isArray(files) ? files : [];
                if (incomingParseStatus && typeof incomingParseStatus === 'object') {
                    parseStatus = incomingParseStatus;
                }
                if (selectedFile && !activeFiles.some((file) => String((file && file.name) || '') === selectedFile)) {
                    selectedFile = null;
                    selectedEmbeddingStatus = null;
                }
                applyMaterialContext(context);
                syncMaterialControlsState();
                renderFiles(activeFiles, config.nofiles || 'No active materials loaded.', openFilePreview, selectedFile);
                renderPreviewEmbeddingStatus(selectedEmbeddingStatus);
                if (!activeFiles.length) {
                    resetPreview(config.previewempty || 'No preview available.');
                    setStatus(config.statusnodocs || 'No materials selected');
                    return;
                }
                setStatus(materialContext.is_manual
                    ? (config.manualmodeactive || 'Manual upload is active. Clear uploaded materials to use class/topic materials again.')
                    : (config.statusready || 'RAG ready'));
            };

            const loadActiveMaterials = async () => {
                const form = new FormData();
                form.append('action', 'list_files');
                form.append('sesskey', config.sesskey);

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload || !payload.ok) {
                        throw new Error((payload && payload.error) || (config.chaterror || 'Failed to load materials.'));
                    }
                    refreshMaterialView(payload.files, payload.material_context, payload.parse_status);
                } catch (err) {
                    parseStatus = {status: 'no_materials', is_parsed: false, parsed_at: 0, sources: 0};
                    renderFiles([], config.nofiles || 'No active materials loaded.', () => {}, selectedFile);
                    resetPreview(config.previewerror || 'Failed to generate preview.');
                    setStatus(config.chaterror || 'Failed to process chat request.');
                }
            };

            const loadMaterialContext = async () => {
                const courseid = chatClassInput ? String(chatClassInput.value || '').trim() : '';
                const topic = chatTopicInput ? String(chatTopicInput.value || '').trim() : '';

                if (materialContext.is_manual) {
                    syncMaterialControlsState();
                    return;
                }

                selectedFile = null;
                selectedEmbeddingStatus = null;
                activeFiles = [];
                renderFiles([], config.nofiles || 'No active materials loaded.', () => {}, selectedFile);
                resetPreview(config.previewempty || 'No preview available.');

                if (!courseid || !topic) {
                    setStatus(config.statusnodocs || 'No materials selected');
                    return;
                }

                setStatus(config.previewloading || 'Loading...');
                const form = new FormData();
                form.append('action', 'set_material_context');
                form.append('sesskey', config.sesskey);
                form.append('courseid', courseid);
                form.append('topic', topic);

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload || !payload.ok) {
                        throw new Error((payload && payload.error) || (config.chaterror || 'Failed to load materials.'));
                    }
                    refreshMaterialView(payload.files, payload.material_context, payload.parse_status);
                } catch (err) {
                    setStatus(config.chaterror || 'Failed to process chat request.');
                    renderFiles([], config.nofiles || 'No active materials loaded.', () => {}, selectedFile);
                    resetPreview(config.previewerror || 'Failed to generate preview.');
                }
            };

            const uploadMaterials = async () => {
                if (!uploadInput || !uploadInput.files || !uploadInput.files.length) {
                    window.alert(config.manualuploadrequired || 'Choose at least one PDF or TXT file first.');
                    return;
                }

                if (uploadBtn) {
                    uploadBtn.disabled = true;
                }
                if (clearUploadBtn) {
                    clearUploadBtn.disabled = true;
                }
                setStatus(config.manualuploading || 'Uploading materials...');

                const form = new FormData();
                form.append('action', 'upload');
                form.append('sesskey', config.sesskey);
                Array.from(uploadInput.files).forEach((file) => {
                    form.append('documents[]', file);
                });

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload || !payload.ok) {
                        throw new Error((payload && payload.error) || (config.chaterror || 'Failed to process chat request.'));
                    }
                    if (chatClassInput) {
                        chatClassInput.value = '';
                    }
                    if (chatTopicInput) {
                        chatTopicInput.innerHTML = '';
                        const placeholder = document.createElement('option');
                        placeholder.value = '';
                        placeholder.textContent = config.coursetopicplaceholder || 'Select topic';
                        chatTopicInput.appendChild(placeholder);
                        chatTopicInput.value = '';
                    }
                    if (uploadInput) {
                        uploadInput.value = '';
                    }
                    selectedFile = null;
                    selectedEmbeddingStatus = null;
                    refreshMaterialView(payload.files, payload.material_context, payload.parse_status);
                } catch (err) {
                    setStatus(err && err.message ? err.message : (config.chaterror || 'Failed to process chat request.'));
                } finally {
                    if (uploadBtn) {
                        uploadBtn.disabled = false;
                    }
                    syncMaterialControlsState();
                }
            };

            const clearUploadedMaterials = async () => {
                if (!materialContext.is_manual) {
                    return;
                }

                if (uploadBtn) {
                    uploadBtn.disabled = true;
                }
                if (clearUploadBtn) {
                    clearUploadBtn.disabled = true;
                }

                const form = new FormData();
                form.append('action', 'clear_uploaded_materials');
                form.append('sesskey', config.sesskey);

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload || !payload.ok) {
                        throw new Error((payload && payload.error) || (config.chaterror || 'Failed to process chat request.'));
                    }
                    selectedFile = null;
                    selectedEmbeddingStatus = null;
                    refreshMaterialView(payload.files, payload.material_context, payload.parse_status);
                    setStatus(config.manualcleared || 'Manual uploaded materials cleared. Topic selection is enabled again.');
                } catch (err) {
                    setStatus(err && err.message ? err.message : (config.chaterror || 'Failed to process chat request.'));
                } finally {
                    if (uploadBtn) {
                        uploadBtn.disabled = false;
                    }
                    syncMaterialControlsState();
                }
            };

            const refreshSelectedEmbedding = async () => {
                const fallbackFile = Array.isArray(activeFiles) && activeFiles.length
                    ? String((activeFiles[0] && activeFiles[0].name) || '').trim()
                    : '';
                const targetFile = getActivePreviewFilename() || fallbackFile;
                if (!targetFile) {
                    setStatus(config.refreshembeddingrequired || 'Load materials first.');
                    return;
                }

                isRefreshEmbeddingBusy = true;
                syncMaterialControlsState();
                setStatus(config.refreshembeddingloading || 'Refreshing embedding index...');

                const form = new FormData();
                form.append('action', 'refresh_selected_embedding');
                form.append('sesskey', config.sesskey);
                form.append('filename', targetFile);

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload || !payload.ok) {
                        throw new Error((payload && payload.error) || (config.refreshembeddingerror || 'Failed to refresh embedding index.'));
                    }
                    selectedEmbeddingStatus = payload.embedding_status || null;
                    refreshMaterialView(payload.files, payload.material_context, payload.parse_status);
                    renderPreviewEmbeddingStatus(selectedEmbeddingStatus);
                    const embeddingSuffix = payload.embedding_model
                        ? `, embedding=${payload.embedding_model}`
                        : '';
                    setStatus(
                        `${config.refreshembeddingok || 'Embedding index refreshed for the active corpus.'} `
                        + `file=${payload.filename || targetFile}, sources=${payload.sources ?? '-'}${embeddingSuffix}`
                    );
                } catch (err) {
                    setStatus(err && err.message ? err.message : (config.refreshembeddingerror || 'Failed to refresh embedding index.'));
                } finally {
                    isRefreshEmbeddingBusy = false;
                    syncMaterialControlsState();
                }
            };

            const openFilePreview = async (filename) => {
                selectedFile = filename;
                selectedEmbeddingStatus = null;
                renderFiles(activeFiles, config.nofiles || 'No active materials loaded.', openFilePreview, selectedFile);
                if (previewName) {
                    previewName.textContent = filename;
                }
                syncMaterialControlsState();
                renderPreviewEmbeddingStatus();
                if (previewBody) {
                    previewBody.innerHTML = `<p class="local-chatbot-empty">${config.previewloading || 'Loading preview...'}</p>`;
                }

                const form = new FormData();
                form.append('action', 'file_content');
                form.append('sesskey', config.sesskey);
                form.append('filename', filename);

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload || !payload.ok) {
                        throw new Error((payload && payload.error) || (config.previewerror || 'Failed to generate preview.'));
                    }

                    selectedEmbeddingStatus = payload.embedding_status || null;
                    renderPreviewEmbeddingStatus(selectedEmbeddingStatus);

                    if (!previewBody) {
                        return;
                    }

                    if (payload.filetype === 'pdf' && payload.viewurl) {
                        previewBody.innerHTML = '';
                        const frame = document.createElement('iframe');
                        frame.src = payload.viewurl;
                        frame.title = filename;
                        frame.loading = 'lazy';
                        previewBody.appendChild(frame);

                        const note = document.createElement('p');
                        note.className = 'local-chatbot-preview-note';
                        note.textContent = String(
                            config.previewpdffallback || 'If the PDF does not render here, open it in a new tab.'
                        );
                        previewBody.appendChild(note);

                        const link = document.createElement('a');
                        link.href = payload.viewurl;
                        link.target = '_blank';
                        link.rel = 'noopener noreferrer';
                        link.textContent = String(config.previewopenpdf || 'Open PDF in new tab');
                        previewBody.appendChild(link);
                        return;
                    }

                    const pre = document.createElement('pre');
                    pre.textContent = String(payload.content || '');
                    previewBody.innerHTML = '';
                    previewBody.appendChild(pre);
                } catch (err) {
                    selectedEmbeddingStatus = null;
                    renderPreviewEmbeddingStatus();
                    resetPreview(config.previewerror || 'Failed to generate preview.');
                }
            };

            const sendMessage = async () => {
                const question = input ? String(input.value || '').trim() : '';
                if (!question) {
                    return;
                }
                const conversationContext = buildConversationContext(history);

                history.push(normalizeHistoryEntry({type: 'user', text: question, sources: []}));
                persistHistory();
                renderHistory();
                input.value = '';

                const placeholder = appendMessageDom({
                    type: 'assistant',
                    text: config.thinking || 'Thinking...',
                    sources: []
                }, config, {});
                isChatBusy = true;
                syncEvaluationSourceState();

                const form = new FormData();
                form.append('action', 'chat');
                form.append('sesskey', config.sesskey);
                form.append('question', question);
                form.append('history', JSON.stringify(conversationContext));
                form.append('chat_mode', getPrimaryMode());
                if (!materialContext.is_manual && chatClassInput && String(chatClassInput.value || '').trim() !== '') {
                    form.append('courseid', String(chatClassInput.value || '').trim());
                }
                if (!materialContext.is_manual && chatTopicInput && String(chatTopicInput.value || '').trim() !== '') {
                    form.append('topic', String(chatTopicInput.value || '').trim());
                }

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload || !payload.ok) {
                        throw new Error((payload && payload.error) || (config.chaterror || 'Failed to process chat request.'));
                    }

                    if (placeholder && placeholder.parentNode) {
                        placeholder.parentNode.removeChild(placeholder);
                    }

                    const answer = String(payload.answer || '');
                    const sources = Array.isArray(payload.sources) ? payload.sources : [];
                    const nextEntry = normalizeHistoryEntry({
                        type: 'assistant',
                        text: answer,
                        sources: sources,
                        requestId: String(payload.request_id || ''),
                        chatMode: String(payload.chat_mode || ''),
                        questionId: String(payload.question_id || ''),
                        runId: Number(payload.run_id || 0),
                        questionText: question,
                        courseId: materialContext.is_manual ? 0 : Number(chatClassInput ? chatClassInput.value || 0 : 0),
                        topic: materialContext.is_manual ? '' : String(chatTopicInput ? chatTopicInput.value || '' : '')
                    });
                    history.push(nextEntry);
                    persistHistory();
                    renderHistory();
                } catch (err) {
                    if (placeholder) {
                        placeholder.textContent = err && err.message ? err.message : (config.chaterror || 'Failed to process chat request.');
                    } else {
                        history.push(normalizeHistoryEntry({
                            type: 'assistant',
                            text: err && err.message ? err.message : (config.chaterror || 'Failed to process chat request.'),
                            sources: []
                        }));
                        persistHistory();
                        renderHistory();
                    }
                } finally {
                    isChatBusy = false;
                    syncEvaluationSourceState();
                    if (input) {
                        if (!input.disabled) {
                            input.focus();
                        }
                    }
                }
            };

            const runEvaluationDataset = async () => {
                if (!evalDatasetFileInput || !evalDatasetFileInput.files || !evalDatasetFileInput.files.length) {
                    window.alert(config.evaldatasetlabel || 'Please choose an evaluation JSON file first.');
                    return;
                }
                if (evalModeInput && evalModeInput.checked && getEvaluationSource() !== 'dataset') {
                    return;
                }
                const selectedModes = getSelectedModes();
                if (!selectedModes.length) {
                    window.alert(config.modellabel || 'Please choose at least one mode first.');
                    return;
                }
                const datasetFile = evalDatasetFileInput.files[0];
                const runsPerQuestion = evalDatasetRunsInput ? String(evalDatasetRunsInput.value || '1').trim() : '1';

                isDatasetBusy = true;
                syncEvaluationSourceState();
                setStatus(config.evaldatasetrunning || 'Running evaluation dataset...');

                const form = new FormData();
                form.append('action', 'run_eval_dataset');
                form.append('sesskey', config.sesskey);
                form.append('dataset', datasetFile);
                form.append('runs_per_question', runsPerQuestion);
                form.append('chat_modes', selectedModes.join(','));
                if (!materialContext.is_manual && chatClassInput && String(chatClassInput.value || '').trim() !== '') {
                    form.append('courseid', String(chatClassInput.value || '').trim());
                }
                if (!materialContext.is_manual && chatTopicInput && String(chatTopicInput.value || '').trim() !== '') {
                    form.append('topic', String(chatTopicInput.value || '').trim());
                }

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload || !payload.ok) {
                        throw new Error((payload && payload.error) || (config.chaterror || 'Failed to process chat request.'));
                    }
                    const summary = payload.summary || {};
                    const objectiveEval = summary.objective_evaluation || {};
                    const objectiveSummary = objectiveEval.summary || {};
                    const modeSummaries = Array.isArray(objectiveSummary.by_mode) ? objectiveSummary.by_mode : [];
                    let message =
                        `${config.evaldatasetsuccess || 'Evaluation dataset finished.'} `
                        + `runs=${summary.total_runs || 0}, success=${summary.successes || 0}, `
                        + `failures=${summary.failures || 0}, file=${summary.output_file || '-'}`;
                    if (modeSummaries.length > 0) {
                        const modeSummaryText = modeSummaries
                            .map((item) => {
                                const modeName = String(item.mode || '-');
                                return `${modeName}[sr=${item.success_rate ?? '-'}, lat=${item.avg_latency_total ?? '-'}, det=${item.answerable_detection_accuracy ?? '-'}, ref=${item.refusal_accuracy ?? '-'}]`;
                            })
                            .join(' ');
                        message += ` | ${modeSummaryText}`;
                    }
                    if (objectiveEval.summary_output_file || objectiveEval.per_run_output_file) {
                        message +=
                            ` | objective_summary=${objectiveEval.summary_output_file || '-'}`
                            + `, objective_runs=${objectiveEval.per_run_output_file || '-'}`;
                    }
                    if (objectiveEval.error) {
                        message += ` | objective_eval_error=${objectiveEval.error}`;
                    }
                    const entry = normalizeHistoryEntry({type: 'assistant', text: message, sources: []});
                    if (appOwnsChat) {
                        appendOwnedChatEntry(entry);
                    } else {
                        appendMessageDom(entry, config, {});
                        history.push(entry);
                        persistHistory();
                    }
                    setStatus(message);
                } catch (err) {
                    const message = err && err.message ? err.message : (config.chaterror || 'Failed to process chat request.');
                    const entry = normalizeHistoryEntry({type: 'assistant', text: message, sources: []});
                    if (appOwnsChat) {
                        appendOwnedChatEntry(entry);
                    } else {
                        appendMessageDom(entry, config, {});
                        history.push(entry);
                        persistHistory();
                    }
                    setStatus(message);
                } finally {
                    isDatasetBusy = false;
                    syncEvaluationSourceState();
                }
            };

            if (chatClassInput && !appOwnsMaterialsPreview) {
                chatClassInput.addEventListener('change', () => {
                    populateTopicOptions(String(chatClassInput.value || '').trim());
                    if (chatTopicInput) {
                        chatTopicInput.value = '';
                    }
                    loadMaterialContext();
                });
            }

            if (chatTopicInput && !appOwnsMaterialsPreview) {
                chatTopicInput.addEventListener('change', () => {
                    loadMaterialContext();
                });
            }

            if (sendBtn && !appOwnsChat) {
                sendBtn.addEventListener('click', sendMessage);
            }

            if (input && !appOwnsChat) {
                input.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault();
                        sendMessage();
                    }
                });
            }

            if (clearBtn && !appOwnsChat) {
                clearBtn.addEventListener('click', () => {
                    if (!window.confirm(config.clearhistoryconfirm || 'Clear this chat history?')) {
                        return;
                    }
                    history = [];
                    persistHistory();
                    renderHistory();
                });
            }

            if (uploadBtn && !appOwnsMaterialsPreview) {
                uploadBtn.addEventListener('click', uploadMaterials);
            }

            if (clearUploadBtn && !appOwnsMaterialsPreview) {
                clearUploadBtn.addEventListener('click', clearUploadedMaterials);
            }
            if (refreshEmbeddingBtn && !appOwnsMaterialsPreview) {
                refreshEmbeddingBtn.addEventListener('click', refreshSelectedEmbedding);
            }

            if (evalModeInput) {
                evalModeInput.addEventListener('change', () => {
                    syncEvalControlsVisibility();
                    syncEvaluationSourceState();
                    renderEmbeddingConfig();
                });
            }

            if (evalSourceInputs.length) {
                evalSourceInputs.forEach((entry) => {
                    entry.addEventListener('change', () => {
                        enforceModeSelectionRules();
                        syncEvaluationSourceState();
                        renderEmbeddingConfig();
                    });
                });
            }

            if (evalDatasetRunBtn) {
                evalDatasetRunBtn.addEventListener('click', runEvaluationDataset);
            }

            if (modeInputs.length) {
                modeInputs.forEach((entry) => {
                    entry.addEventListener('change', () => {
                        enforceModeSelectionRules(entry);
                        renderEmbeddingConfig();
                    });
                });
            }

            if (!appOwnsMaterialsPreview && previewName && typeof MutationObserver !== 'undefined') {
                const previewNameObserver = new MutationObserver(() => {
                    renderPreviewEmbeddingStatus(selectedEmbeddingStatus);
                    syncMaterialControlsState();
                });
                previewNameObserver.observe(previewName, {
                    childList: true,
                    characterData: true,
                    subtree: true
                });
            }

            if (appOwnsMaterialsPreview && appRoot) {
                const onMaterialsSync = (event) => {
                    syncMaterialsFromAppDetail((event && event.detail) || {});
                };
                appRoot.addEventListener('local-chatbot:materials-sync', onMaterialsSync);
                if (appRoot.__localChatbotApp && appRoot.__localChatbotApp.store) {
                    syncMaterialsFromAppState(appRoot.__localChatbotApp.store.getState());
                }
            }

            syncEvalControlsVisibility();
            syncEvaluationSourceState();
            renderEmbeddingConfig();

            if (!appOwnsMaterialsPreview) {
                syncMaterialControlsState();
                populateTopicOptions(chatClassInput ? String(chatClassInput.value || '').trim() : '');
                renderFiles([], config.nofiles || 'No active materials loaded.', () => {}, selectedFile);
                resetPreview(config.previewempty || 'No preview available.');
                setStatus(config.statusnodocs || 'No materials selected');
            }
            if (!appOwnsChat) {
                persistHistory();
                renderHistory();
            }
            if (!appOwnsMaterialsPreview) {
                loadActiveMaterials();
            }
        }
    };
});
