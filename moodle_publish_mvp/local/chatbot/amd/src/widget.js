define(['core/log'], function(Log) {
    const MAX_HISTORY = 80;
    const MAX_USER_MESSAGES = 80;
    const MAX_QUESTION_COMPONENTS = 10;

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
            // Ignore storage write errors (private mode/quota).
        }
    };

    const userMessageCount = (history) => history.filter((entry) => entry.type === 'user').length;

    const trimByUserLimit = (history) => {
        const next = Array.isArray(history) ? history.slice() : [];
        while (userMessageCount(next) > MAX_USER_MESSAGES && next.length > 0) {
            next.shift();
        }
        return next.slice(-MAX_HISTORY);
    };

    const renderFiles = (files, nofiles, onFileClick, selectedFile) => {
        const wrap = document.getElementById('local-chatbot-files');
        const status = document.getElementById('local-chatbot-status');
        if (!wrap) {
            return;
        }

        if (!files || files.length === 0) {
            wrap.innerHTML = `<p class="local-chatbot-empty">${nofiles}</p>`;
            if (status) {
                status.textContent = 'No materials selected';
            }
            return;
        }

        wrap.innerHTML = '';
        files.forEach((file) => {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'local-chatbot-file-item';
            if (selectedFile && selectedFile === file.name) {
                item.classList.add('active');
            }
            item.dataset.file = file.name;
            item.innerHTML = `<span>${file.name}</span>`;
            if (typeof onFileClick === 'function') {
                item.addEventListener('click', () => onFileClick(file.name));
            }
            wrap.appendChild(item);
        });
        if (status) {
            status.textContent = 'RAG ready';
        }
    };

    const appendMessageDom = (text, type, sources, asHtml = false) => {
        const messages = document.getElementById('local-chatbot-messages');
        if (!messages) {
            return null;
        }

        const item = document.createElement('div');
        item.className = `local-chatbot-message ${type}`;
        if (asHtml) {
            item.innerHTML = String(text || '');
        } else {
            item.textContent = text;
        }
        messages.appendChild(item);

        if (sources && sources.length > 0) {
            const source = document.createElement('div');
            source.className = 'local-chatbot-source';
            source.textContent = `source: ${sources.join(', ')}`;
            item.appendChild(source);
        }

        messages.scrollTop = messages.scrollHeight;
        return item;
    };

    const postForm = async (url, data, options = {}) => {
        const res = await fetch(url, {
            method: 'POST',
            body: data,
            signal: options.signal,
            credentials: 'same-origin',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        });
        const raw = await res.text();
        if (!raw || !raw.trim()) {
            throw new Error(`Empty response from server (HTTP ${res.status}).`);
        }
        try {
            return JSON.parse(raw);
        } catch (err) {
            const snippet = raw.slice(0, 180).replace(/\s+/g, ' ').trim();
            throw new Error(`Invalid JSON response (HTTP ${res.status}): ${snippet || 'no payload'}`);
        }
    };

    const formatHistoryTime = (value) => {
        if (!value) {
            return '-';
        }
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) {
            return '-';
        }
        return date.toLocaleString();
    };

    return {
        init: function(config) {
            Log.debug('local_chatbot page initialized');

            const input = document.getElementById('local-chatbot-input');
            const sendBtn = document.getElementById('local-chatbot-send');
            const clearBtn = document.getElementById('local-chatbot-clear');
            const messagesWrap = document.getElementById('local-chatbot-messages');
            const usageWrap = document.getElementById('local-chatbot-usage');
            const previewBody = document.getElementById('local-chatbot-preview-body');
            const previewName = document.getElementById('local-chatbot-preview-name');
            const historyWrap = document.getElementById('local-chatbot-history-list');
            const tabButtons = document.querySelectorAll('.local-chatbot-tab');
            const chatClassInput = document.getElementById('local-chatbot-chat-class');
            const chatTopicInput = document.getElementById('local-chatbot-chat-topic');
            const chatMasteryBadge = document.getElementById('local-chatbot-mastery');

            const assignClassInput = document.getElementById('local-chatbot-assign-class');
            const assignTopicInput = document.getElementById('local-chatbot-assign-topic');
            const assignPdfInput = document.getElementById('local-chatbot-assign-pdf');
            const assignTypeInput = document.getElementById('local-chatbot-assign-type');
            const assignTypeCustomWrap = document.getElementById('local-chatbot-assign-type-custom-wrap');
            const assignTypeCustomInput = document.getElementById('local-chatbot-assign-type-custom');
            const assignWeightLabelInput = document.getElementById('local-chatbot-assign-weight-label');
            const assignWeightPercentInput = document.getElementById('local-chatbot-assign-weight-percent');
            const assignCountWrap = document.getElementById('local-chatbot-assign-count-wrap');
            const assignCountInput = document.getElementById('local-chatbot-assign-count');
            const assignNotesInput = document.getElementById('local-chatbot-assign-notes');
            const assignGenerateBtn = document.getElementById('local-chatbot-assign-generate');
            const assignRegenerateBtn = document.getElementById('local-chatbot-assign-regenerate');
            const assignPublishBtn = document.getElementById('local-chatbot-assign-publish');
            const assignPreview = document.getElementById('local-chatbot-assign-preview');
            const assignMasteryBadge = document.getElementById('local-chatbot-assign-mastery');

            const practiceClassInput = document.getElementById('local-chatbot-practice-class');
            const practiceTopicInput = document.getElementById('local-chatbot-practice-topic');
            const practicePdfInput = document.getElementById('local-chatbot-practice-pdf');
            const practiceCountInput = document.getElementById('local-chatbot-practice-count');
            const practicePageStartInput = document.getElementById('local-chatbot-practice-page-start');
            const practicePageEndInput = document.getElementById('local-chatbot-practice-page-end');
            const practiceGenerateBtn = document.getElementById('local-chatbot-practice-generate');
            const practiceStopBtn = document.getElementById('local-chatbot-practice-stop');
            const practicePrepareBtn = document.getElementById('local-chatbot-practice-prepare');
            const practicePublishBtn = document.getElementById('local-chatbot-practice-publish');
            const practiceRetryQuestionBtn = document.getElementById('local-chatbot-practice-retry-question');
            const practiceContinuePublishBtn = document.getElementById('local-chatbot-practice-continue-publish');
            const practicePreview = document.getElementById('local-chatbot-practice-preview');
            const practiceMasteryBadge = document.getElementById('local-chatbot-practice-mastery');

            const storageKey = `local_chatbot_history_u${config.userid || 'anon'}`;
            const configuredCourseTopics = (config && typeof config.coursetopics === 'object' && config.coursetopics !== null)
                ? config.coursetopics
                : {};
            const configuredCoursePdfs = (config && typeof config.coursepdfs === 'object' && config.coursepdfs !== null)
                ? config.coursepdfs
                : {};

            let history = safeReadHistory(storageKey);
            let selectedFile = null;
            const masteryLabel = String(config.chatmasterylabel || 'Mastery');
            const taskMasteryLabel = String(config.taskmasterylabel || 'Topic mastery');
            const taskMasteryUnknown = String(config.taskmasteryunknown || 'Topic mastery: select class and topic');
            const taskMasteryLoading = String(config.taskmasteryloading || 'Topic mastery: loading...');
            const taskMasteryNoData = String(config.taskmasterynodata || 'Topic mastery: no data yet');

            const setMasteryBadge = (text, tone = 'neutral') => {
                if (!chatMasteryBadge) {
                    return;
                }
                chatMasteryBadge.textContent = String(text || '');
                const tones = ['neutral', 'loading', 'empty', 'low', 'mid', 'high'];
                tones.forEach((item) => {
                    chatMasteryBadge.classList.remove(`local-chatbot-mastery--${item}`);
                });
                const safeTone = tones.includes(tone) ? tone : 'neutral';
                chatMasteryBadge.classList.add(`local-chatbot-mastery--${safeTone}`);
            };

            const formatMasteryPercent = (value) => {
                const numeric = Number(value);
                if (!Number.isFinite(numeric)) {
                    return null;
                }
                return numeric.toFixed(1);
            };

            const applyMasteryContext = (context) => {
                if (!context || typeof context !== 'object') {
                    setMasteryBadge(config.chatmasteryunknown || 'Mastery: select class and topic', 'neutral');
                    return;
                }

                const group = String(context.group || 'mid').trim().toLowerCase();
                const groupLabel = ['low', 'mid', 'high'].includes(group) ? group.toUpperCase() : 'MID';
                const masteryText = formatMasteryPercent(context.mastery);
                if (masteryText !== null) {
                    setMasteryBadge(`${masteryLabel}: ${masteryText}% (${groupLabel})`, group);
                    return;
                }

                const status = String(context.status || '').trim().toLowerCase();
                if (status === 'no_topic_mastery_data') {
                    setMasteryBadge(`${config.chatmasterynodata || 'Mastery: no data yet'} (${groupLabel})`, 'empty');
                    return;
                }
                setMasteryBadge(`${masteryLabel}: - (${groupLabel})`, 'neutral');
            };

            const setTopicMasteryBadge = (badgeElement, text, tone = 'neutral') => {
                if (!badgeElement) {
                    return;
                }
                badgeElement.textContent = String(text || '');
                const tones = ['neutral', 'loading', 'empty', 'low', 'mid', 'high'];
                tones.forEach((item) => {
                    badgeElement.classList.remove(`local-chatbot-topic-mastery--${item}`);
                });
                const safeTone = tones.includes(tone) ? tone : 'neutral';
                badgeElement.classList.add(`local-chatbot-topic-mastery--${safeTone}`);
            };

            const applyTaskMasteryContext = (badgeElement, context) => {
                if (!badgeElement) {
                    return;
                }
                if (!context || typeof context !== 'object') {
                    setTopicMasteryBadge(badgeElement, taskMasteryUnknown, 'neutral');
                    return;
                }

                const group = String(context.group || 'mid').trim().toLowerCase();
                const groupLabel = ['low', 'mid', 'high'].includes(group) ? group.toUpperCase() : 'MID';
                const masteryText = formatMasteryPercent(context.mastery);
                if (masteryText !== null) {
                    setTopicMasteryBadge(badgeElement, `${taskMasteryLabel}: ${masteryText}% (${groupLabel})`, group);
                    return;
                }

                const status = String(context.status || '').trim().toLowerCase();
                if (status === 'no_topic_mastery_data') {
                    setTopicMasteryBadge(badgeElement, `${taskMasteryNoData} (${groupLabel})`, 'empty');
                    return;
                }
                setTopicMasteryBadge(badgeElement, `${taskMasteryLabel}: - (${groupLabel})`, 'neutral');
            };

            const fetchTopicMasteryContext = async (courseidRaw, courseName, topic) => {
                const numericCourseId = /^\d+$/.test(String(courseidRaw || '').trim())
                    ? String(courseidRaw || '').trim()
                    : '0';
                const form = new FormData();
                form.append('action', 'topic_mastery');
                form.append('sesskey', config.sesskey);
                form.append('courseid', numericCourseId);
                form.append('course_name', String(courseName || '').trim());
                form.append('topic', String(topic || '').trim());
                const payload = await postForm(config.ajaxurl, form);
                if (!payload || !payload.ok) {
                    // Fallback: reuse set_material_context flow that already returns mastery_context.
                    const fallbackForm = new FormData();
                    fallbackForm.append('action', 'set_material_context');
                    fallbackForm.append('sesskey', config.sesskey);
                    fallbackForm.append('courseid', numericCourseId);
                    fallbackForm.append('course_name', String(courseName || '').trim());
                    fallbackForm.append('topic', String(topic || '').trim());
                    const fallbackPayload = await postForm(config.ajaxurl, fallbackForm);
                    if (!fallbackPayload || !fallbackPayload.ok) {
                        throw new Error((payload && payload.error) ? payload.error : 'Failed to load mastery');
                    }
                    return fallbackPayload.mastery_context || null;
                }
                return payload.mastery_context || null;
            };

            const updateAssignmentMastery = async (courseidRaw, courseName, topic) => {
                if (!assignMasteryBadge) {
                    return;
                }
                if ((!String(courseidRaw || '').trim() && !String(courseName || '').trim()) || !String(topic || '').trim()) {
                    setTopicMasteryBadge(assignMasteryBadge, taskMasteryUnknown, 'neutral');
                    return;
                }
                setTopicMasteryBadge(assignMasteryBadge, taskMasteryLoading, 'loading');
                try {
                    const context = await fetchTopicMasteryContext(courseidRaw, courseName, topic);
                    applyTaskMasteryContext(assignMasteryBadge, context);
                } catch (err) {
                    setTopicMasteryBadge(assignMasteryBadge, taskMasteryUnknown, 'neutral');
                }
            };

            const updatePracticeMastery = async (courseidRaw, courseName, topic) => {
                if (!practiceMasteryBadge) {
                    return;
                }
                if ((!String(courseidRaw || '').trim() && !String(courseName || '').trim()) || !String(topic || '').trim()) {
                    setTopicMasteryBadge(practiceMasteryBadge, taskMasteryUnknown, 'neutral');
                    return;
                }
                setTopicMasteryBadge(practiceMasteryBadge, taskMasteryLoading, 'loading');
                try {
                    const context = await fetchTopicMasteryContext(courseidRaw, courseName, topic);
                    applyTaskMasteryContext(practiceMasteryBadge, context);
                } catch (err) {
                    setTopicMasteryBadge(practiceMasteryBadge, taskMasteryUnknown, 'neutral');
                }
            };
            let assignmentLastPrompt = '';
            let assignmentLastDraftText = '';
            let practiceLastPrompt = '';
            let practiceLastDraftText = '';
            let practiceRawDraftText = '';
            let practicePreparedDraftText = '';
            let practiceIsPreparedForPublish = false;
            let practiceStopRequested = false;
            let practiceGenerationInProgress = false;
            let practiceActiveAbortController = null;
            let practiceGenerationState = {
                total: 0,
                generatedQuestions: [],
                blocked: false,
                failedQuestionNumber: 0,
                lastErrorMessage: ''
            };

            const isAbortError = (err) => {
                if (!err) {
                    return false;
                }
                const name = String(err.name || '').trim();
                const message = String(err.message || '').toLowerCase().trim();
                return name === 'AbortError' || message.includes('aborted');
            };

            const isPracticeStopError = (err) => {
                const message = String(err && err.message ? err.message : '').toLowerCase();
                return message.includes('stopped by user');
            };

            const generateRequestId = () => {
                const now = Date.now();
                if (window.crypto && typeof window.crypto.randomUUID === 'function') {
                    return `req-${now}-${window.crypto.randomUUID()}`;
                }
                const random = Math.random().toString(36).slice(2, 10);
                return `req-${now}-${random}`;
            };

            const setPracticeStopButtonState = (inProgress) => {
                if (!practiceStopBtn) {
                    return;
                }
                practiceStopBtn.disabled = !inProgress;
            };

            const resetPracticePreparedState = () => {
                practicePreparedDraftText = '';
                practiceIsPreparedForPublish = false;
            };

            const resetPracticeGenerationState = () => {
                practiceStopRequested = false;
                practiceGenerationInProgress = false;
                if (practiceActiveAbortController) {
                    practiceActiveAbortController.abort();
                }
                practiceActiveAbortController = null;
                practiceGenerationState = {
                    total: 0,
                    generatedQuestions: [],
                    blocked: false,
                    failedQuestionNumber: 0,
                    lastErrorMessage: ''
                };
                if (practiceRetryQuestionBtn) {
                    practiceRetryQuestionBtn.hidden = true;
                    practiceRetryQuestionBtn.disabled = true;
                }
                if (practiceContinuePublishBtn) {
                    practiceContinuePublishBtn.hidden = true;
                    practiceContinuePublishBtn.disabled = true;
                }
                setPracticeStopButtonState(false);
            };

            const setPracticeRecoveryActionsVisible = (visible) => {
                if (practiceRetryQuestionBtn) {
                    practiceRetryQuestionBtn.hidden = !visible;
                    practiceRetryQuestionBtn.disabled = !visible;
                }
                if (practiceContinuePublishBtn) {
                    practiceContinuePublishBtn.hidden = !visible;
                    practiceContinuePublishBtn.disabled = !visible;
                }
            };

            const setPracticeGenerationControlsDisabled = (disabled) => {
                if (practiceGenerateBtn) {
                    practiceGenerateBtn.disabled = disabled;
                }
                if (practicePrepareBtn) {
                    practicePrepareBtn.disabled = disabled;
                }
                if (practicePublishBtn) {
                    practicePublishBtn.disabled = disabled;
                }
                if (practiceRetryQuestionBtn) {
                    practiceRetryQuestionBtn.disabled = disabled || practiceRetryQuestionBtn.hidden;
                }
                if (practiceContinuePublishBtn) {
                    practiceContinuePublishBtn.disabled = disabled || practiceContinuePublishBtn.hidden;
                }
            };

            const panelMap = {
                chat: document.getElementById('local-chatbot-panel-chat'),
                assignment: document.getElementById('local-chatbot-panel-assignment'),
                practice: document.getElementById('local-chatbot-panel-practice'),
                history: document.getElementById('local-chatbot-panel-history')
            };

            const updateUsage = () => {
                if (!usageWrap) {
                    return;
                }
                const used = userMessageCount(history);
                const label = config.chatusagelabel || 'Chat usage';
                usageWrap.textContent = `${label}: ${used}/${MAX_USER_MESSAGES}`;
            };

            const renderHistoryPanel = () => {
                if (!historyWrap) {
                    return;
                }
                historyWrap.innerHTML = '';
                if (!history.length) {
                    const empty = document.createElement('p');
                    empty.className = 'local-chatbot-empty';
                    empty.textContent = config.historyempty || 'No chat history yet.';
                    historyWrap.appendChild(empty);
                    return;
                }

                history.forEach((entry) => {
                    const item = document.createElement('div');
                    item.className = `local-chatbot-history-item ${entry.type === 'user' ? 'user' : 'bot'}`;

                    const meta = document.createElement('div');
                    meta.className = 'local-chatbot-history-meta';
                    const role = entry.type === 'user' ? 'User' : 'Assistant';
                    meta.textContent = `${role} - ${formatHistoryTime(entry.time)}`;

                    const body = document.createElement('div');
                    body.textContent = entry.text || '';

                    item.appendChild(meta);
                    item.appendChild(body);

                    if (Array.isArray(entry.sources) && entry.sources.length > 0) {
                        const src = document.createElement('div');
                        src.className = 'local-chatbot-history-meta';
                        src.textContent = `source: ${entry.sources.join(', ')}`;
                        item.appendChild(src);
                    }
                    historyWrap.appendChild(item);
                });
            };

            const persistHistory = () => {
                history = trimByUserLimit(history);
                safeWriteHistory(storageKey, history);
                updateUsage();
                renderHistoryPanel();
            };

            const appendMessage = (text, type, sources, persist, asHtml = false) => {
                appendMessageDom(text, type, sources, asHtml);
                if (!persist) {
                    return;
                }
                history.push({
                    text: String(text),
                    type: type === 'user' ? 'user' : 'bot',
                    sources: Array.isArray(sources) ? sources : [],
                    time: new Date().toISOString()
                });
                persistHistory();
            };

            const restoreHistory = () => {
                if (!messagesWrap) {
                    return;
                }
                messagesWrap.innerHTML = '';

                if (!history.length) {
                    appendMessage(config.defaultgreeting || 'Hello.', 'bot', [], true);
                    return;
                }

                history.forEach((entry) => {
                    appendMessage(
                        entry.text || '',
                        entry.type === 'user' ? 'user' : 'bot',
                        Array.isArray(entry.sources) ? entry.sources : [],
                        false
                    );
                });
                updateUsage();
                renderHistoryPanel();
            };

            const setActiveFile = (filename) => {
                const items = document.querySelectorAll('.local-chatbot-file-item');
                items.forEach((item) => {
                    if (item.dataset.file === filename) {
                        item.classList.add('active');
                    } else {
                        item.classList.remove('active');
                    }
                });
            };

            const setPreviewEmpty = (text) => {
                if (!previewBody) {
                    return;
                }
                previewBody.innerHTML = '';
                const p = document.createElement('p');
                p.className = 'local-chatbot-empty';
                p.textContent = text;
                previewBody.appendChild(p);
            };

            const switchTab = (tabName) => {
                tabButtons.forEach((button) => {
                    button.classList.toggle('active', button.dataset.tab === tabName);
                });
                Object.keys(panelMap).forEach((key) => {
                    if (!panelMap[key]) {
                        return;
                    }
                    panelMap[key].classList.toggle('active', key === tabName);
                });
                if (tabName === 'history') {
                    renderHistoryPanel();
                }
            };

            const runChatRequest = async (question, options = {}) => {
                const form = new FormData();
                const requestId = String(options.requestId || generateRequestId()).trim();
                const questionNumber = Number.isFinite(Number(options.questionNumber))
                    ? Number(options.questionNumber)
                    : 0;
                const attempt = Number.isFinite(Number(options.attempt))
                    ? Number(options.attempt)
                    : 0;
                const generationMode = String(options.generationMode || '').trim().toLowerCase();
                form.append('action', 'chat');
                form.append('sesskey', config.sesskey);
                form.append('question', question);
                if (requestId) {
                    form.append('request_id', requestId);
                }
                if (questionNumber > 0) {
                    form.append('question_number', String(questionNumber));
                }
                if (attempt > 0) {
                    form.append('generation_attempt', String(attempt));
                }
                if (generationMode) {
                    form.append('generation_mode', generationMode);
                }
                const courseid = String(options.courseid || '').trim();
                const topic = String(options.topic || '').trim();
                const pageStart = Number.isFinite(Number(options.pageStart))
                    ? Number(options.pageStart)
                    : 0;
                const pageEnd = Number.isFinite(Number(options.pageEnd))
                    ? Number(options.pageEnd)
                    : 0;
                if (courseid) {
                    form.append('courseid', courseid);
                }
                if (topic) {
                    form.append('topic', topic);
                }
                if (pageStart > 0) {
                    form.append('page_start', String(pageStart));
                }
                if (pageEnd > 0) {
                    form.append('page_end', String(pageEnd));
                }
                const started = Date.now();
                try {
                    const payload = await postForm(config.ajaxurl, form, { signal: options.signal });
                    const duration = Date.now() - started;
                    Log.debug(`local_chatbot chat request success request_id=${requestId} question=${questionNumber || '-'} attempt=${attempt || '-'} duration_ms=${duration}`);
                    if (!payload.ok) {
                        const backendRequestId = String(payload.request_id || requestId || '').trim();
                        const prefix = backendRequestId ? `[request_id: ${backendRequestId}] ` : '';
                        throw new Error(prefix + (payload.error || config.chaterror));
                    }
                    payload.request_id = payload.request_id || requestId;
                    return payload;
                } catch (err) {
                    const duration = Date.now() - started;
                    const message = err && err.message ? err.message : 'unknown error';
                    Log.debug(`local_chatbot chat request error request_id=${requestId} question=${questionNumber || '-'} attempt=${attempt || '-'} duration_ms=${duration} error=${message}`);
                    throw err;
                }
            };

            const renderMarkdownRequest = async (markdownText) => {
                const form = new FormData();
                form.append('action', 'render_markdown');
                form.append('sesskey', config.sesskey);
                form.append('text', String(markdownText || ''));
                const payload = await postForm(config.ajaxurl, form);
                if (!payload.ok) {
                    throw new Error(payload.error || 'Failed to render markdown');
                }
                return String(payload.html || '');
            };

            const setGeneratedContent = async (target, text, asMarkdown) => {
                if (!target) {
                    return;
                }
                const normalized = String(text || '').trim();
                if (!asMarkdown || normalized === '') {
                    target.textContent = normalized;
                    return;
                }
                try {
                    const html = await renderMarkdownRequest(normalized);
                    if (html.trim() === '') {
                        target.textContent = normalized;
                        return;
                    }
                    target.innerHTML = html;
                } catch (err) {
                    target.textContent = normalized;
                }
            };

            const loadPreview = async (filename) => {
                if (!filename || !previewBody) {
                    return;
                }
                selectedFile = filename;
                if (previewName) {
                    previewName.textContent = filename;
                }
                setActiveFile(filename);
                setPreviewEmpty(config.previewloading || 'Loading preview...');

                const form = new FormData();
                form.append('action', 'file_content');
                form.append('sesskey', config.sesskey);
                form.append('filename', filename);

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload.ok) {
                        throw new Error(payload.error || config.previewerror || 'Preview failed');
                    }

                    previewBody.innerHTML = '';
                    if (payload.filetype === 'pdf' && payload.viewurl) {
                        const iframe = document.createElement('iframe');
                        iframe.src = payload.viewurl;
                        iframe.setAttribute('title', payload.filename || filename);
                        previewBody.appendChild(iframe);
                        return;
                    }

                    const pre = document.createElement('pre');
                    pre.className = 'local-chatbot-preview-text';
                    pre.textContent = payload.content || '';
                    previewBody.appendChild(pre);

                    if (payload.truncated) {
                        const note = document.createElement('p');
                        note.className = 'local-chatbot-preview-note';
                        note.textContent = 'Preview truncated to first 200000 characters.';
                        previewBody.appendChild(note);
                    }
                } catch (err) {
                    setPreviewEmpty(err.message || config.previewerror || 'Failed to load preview');
                }
            };

            const refreshFiles = async (preferredFile) => {
                const form = new FormData();
                form.append('action', 'list_files');
                form.append('sesskey', config.sesskey);

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload.ok) {
                        throw new Error(payload.error || config.nofiles || 'No files');
                    }
                    const files = Array.isArray(payload.files) ? payload.files : [];
                    const nextFile = preferredFile || selectedFile;
                    renderFiles(files, config.nofiles, loadPreview, nextFile);

                    if (!files.length) {
                        selectedFile = null;
                        if (previewName) {
                            previewName.textContent = '-';
                        }
                        setPreviewEmpty(config.previewempty || 'Click a file to preview');
                        return;
                    }

                    let fileToOpen = files[0].name;
                    if (nextFile && files.some((f) => f.name === nextFile)) {
                        fileToOpen = nextFile;
                    }
                    await loadPreview(fileToOpen);
                } catch (err) {
                    setPreviewEmpty(err.message || config.previewerror || 'Failed to load preview');
                }
            };

            const setChatTopicOptions = (topics, placeholderText) => {
                if (!chatTopicInput) {
                    return;
                }
                chatTopicInput.innerHTML = '';
                if (placeholderText) {
                    const placeholder = document.createElement('option');
                    placeholder.value = '';
                    placeholder.textContent = placeholderText;
                    chatTopicInput.appendChild(placeholder);
                }
                if (!Array.isArray(topics)) {
                    return;
                }
                topics.forEach((topic) => {
                    const option = document.createElement('option');
                    option.value = String(topic.value || topic.label || '');
                    option.textContent = String(topic.label || topic.value || '');
                    chatTopicInput.appendChild(option);
                });
                if (chatTopicInput.options.length > 1) {
                    chatTopicInput.selectedIndex = 1;
                } else {
                    chatTopicInput.selectedIndex = 0;
                }
            };

            const loadChatTopics = async () => {
                if (!chatClassInput || !chatTopicInput) {
                    return;
                }
                const courseid = String(chatClassInput.value || '').trim();
                if (!courseid) {
                    setChatTopicOptions([], config.assignmenttopicplaceholder || 'Select class first');
                    return;
                }

                if (Object.prototype.hasOwnProperty.call(configuredCourseTopics, courseid)) {
                    const localTopics = Array.isArray(configuredCourseTopics[courseid]) ? configuredCourseTopics[courseid] : [];
                    if (localTopics.length > 0) {
                        setChatTopicOptions(localTopics, config.assignmenttopicplaceholder || 'Select a topic');
                    } else {
                        setChatTopicOptions([], config.assignmenttopicempty || 'No topics found in this class');
                    }
                    return;
                }

                setChatTopicOptions([], config.assignmenttopicloading || 'Loading topics...');
                const form = new FormData();
                form.append('action', 'course_topics');
                form.append('sesskey', config.sesskey);
                form.append('courseid', courseid);
                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload.ok) {
                        throw new Error(payload.error || config.assignmenttopicempty || 'No topics found');
                    }
                    const topics = Array.isArray(payload.topics) ? payload.topics : [];
                    if (!topics.length) {
                        setChatTopicOptions([], config.assignmenttopicempty || 'No topics found in this class');
                        return;
                    }
                    setChatTopicOptions(topics, config.assignmenttopicplaceholder || 'Select a topic');
                } catch (err) {
                    setChatTopicOptions([], err.message || config.assignmenttopicempty || 'No topics found in this class');
                }
            };

            const syncChatMaterials = async () => {
                if (!chatClassInput || !chatTopicInput) {
                    await refreshFiles(null);
                    return;
                }
                const courseid = String(chatClassInput.value || '').trim();
                const topic = String(chatTopicInput.value || '').trim();

                if (!courseid || !topic) {
                    renderFiles([], config.nofiles, loadPreview, null);
                    selectedFile = null;
                    if (previewName) {
                        previewName.textContent = '-';
                    }
                    setMasteryBadge(config.chatmasteryunknown || 'Mastery: select class and topic', 'neutral');
                    setPreviewEmpty(config.previewempty || 'Click a file to preview');
                    return;
                }

                const selected = chatClassInput.options[chatClassInput.selectedIndex];
                const coursename = selected && selected.dataset && selected.dataset.coursename
                    ? String(selected.dataset.coursename).trim()
                    : String(selected ? selected.text : '').trim();

                const form = new FormData();
                form.append('action', 'set_material_context');
                form.append('sesskey', config.sesskey);
                form.append('courseid', courseid);
                form.append('course_name', coursename);
                form.append('topic', topic);

                try {
                    setMasteryBadge(config.chatmasteryloading || 'Mastery: loading...', 'loading');
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload.ok) {
                        throw new Error(payload.error || config.nofiles || 'No materials found');
                    }
                    applyMasteryContext(payload.mastery_context || null);
                    const files = Array.isArray(payload.files) ? payload.files : [];
                    renderFiles(files, config.nofiles, loadPreview, selectedFile);

                    if (!files.length) {
                        selectedFile = null;
                        if (previewName) {
                            previewName.textContent = '-';
                        }
                        setPreviewEmpty(config.previewempty || 'Click a file to preview');
                        return;
                    }

                    const fileToOpen = (selectedFile && files.some((f) => f.name === selectedFile))
                        ? selectedFile
                        : files[0].name;
                    await loadPreview(fileToOpen);
                } catch (err) {
                    renderFiles([], err.message || config.nofiles, loadPreview, null);
                    selectedFile = null;
                    if (previewName) {
                        previewName.textContent = '-';
                    }
                    setMasteryBadge(config.chatmasteryunknown || 'Mastery: select class and topic', 'neutral');
                    setPreviewEmpty(err.message || config.previewerror || 'Failed to load preview');
                }
            };

            const sendMessage = async () => {
                if (!input) {
                    return;
                }
                const question = input.value.trim();
                if (!question) {
                    return;
                }
                appendMessage(question, 'user', [], true);
                input.value = '';

                const pending = appendMessageDom(config.thinking || 'Thinking...', 'bot', []);

                try {
                    const courseid = chatClassInput ? String(chatClassInput.value || '').trim() : '';
                    const topic = chatTopicInput ? String(chatTopicInput.value || '').trim() : '';
                    const payload = await runChatRequest(question, {
                        courseid: courseid,
                        topic: topic
                    });
                    if (pending) {
                        pending.remove();
                    }
                    const answerText = String(payload.answer || '');
                    let renderedHtml = '';
                    try {
                        renderedHtml = await renderMarkdownRequest(answerText);
                    } catch (renderErr) {
                        renderedHtml = '';
                    }
                    appendMessage(
                        renderedHtml || answerText,
                        'bot',
                        payload.sources || [],
                        true,
                        renderedHtml.trim() !== ''
                    );
                } catch (err) {
                    if (pending) {
                        pending.remove();
                    }
                    appendMessage(err.message || config.chaterror, 'bot', [], true);
                }
            };

            const clearHistory = () => {
                const ask = config.clearhistoryconfirm || 'Clear all messages in this chat history?';
                if (!window.confirm(ask)) {
                    return;
                }
                history = [];
                safeWriteHistory(storageKey, history);
                restoreHistory();
                if (input) {
                    input.focus();
                }
            };

            const clampQuestionCount = (raw) => {
                let count = parseInt(String(raw || '').trim(), 10);
                if (Number.isNaN(count)) {
                    count = 5;
                }
                if (count < 1) {
                    count = 1;
                }
                if (count > MAX_QUESTION_COMPONENTS) {
                    count = MAX_QUESTION_COMPONENTS;
                }
                return count;
            };

            const normalizeQuestionCountInput = (inputElement) => {
                const count = clampQuestionCount(inputElement ? inputElement.value : '5');
                if (inputElement) {
                    inputElement.value = String(count);
                }
                return String(count);
            };

            const parsePositiveInteger = (raw) => {
                const value = parseInt(String(raw || '').trim(), 10);
                if (!Number.isFinite(value) || Number.isNaN(value) || value <= 0) {
                    return 0;
                }
                return value;
            };

            const getPracticePageRange = () => {
                let pageStart = parsePositiveInteger(practicePageStartInput ? practicePageStartInput.value : '');
                let pageEnd = parsePositiveInteger(practicePageEndInput ? practicePageEndInput.value : '');

                if (pageStart > 0 && pageEnd === 0) {
                    pageEnd = pageStart;
                } else if (pageEnd > 0 && pageStart === 0) {
                    pageStart = pageEnd;
                }

                if (pageStart > 0 && pageEnd > 0 && pageEnd < pageStart) {
                    const swapped = pageStart;
                    pageStart = pageEnd;
                    pageEnd = swapped;
                }

                if (practicePageStartInput) {
                    practicePageStartInput.value = pageStart > 0 ? String(pageStart) : '';
                }
                if (practicePageEndInput) {
                    practicePageEndInput.value = pageEnd > 0 ? String(pageEnd) : '';
                }

                return {
                    pageStart,
                    pageEnd
                };
            };

            const resolveAssignmentSelection = () => {
                const selected = assignTypeInput
                    ? String(assignTypeInput.value || 'essay').trim()
                    : 'essay';

                const map = {
                    'essay': {
                        format: 'essay',
                        label: 'Essay Assignment',
                        context: 'General essay assignment for individual student submission.',
                        weighttype: 'individual'
                    },
                    'individual-essay': {
                        format: 'essay',
                        label: 'Individual Assignment',
                        context: 'Designed for individual student submission.',
                        weighttype: 'individual'
                    },
                    'group-essay': {
                        format: 'essay',
                        label: 'Group Assignment',
                        context: 'Designed for collaborative group submission.',
                        weighttype: 'group'
                    },
                    'summary-essay': {
                        format: 'essay',
                        label: 'Summary Assignment',
                        context: 'Focus on concise summary and comprehension of the topic.',
                        weighttype: 'individual'
                    },
                    'presentation-essay': {
                        format: 'essay',
                        label: 'Presentation Assignment',
                        context: 'Answers should support oral presentation and slide delivery.',
                        weighttype: 'individual'
                    },
                    'lab-report-essay': {
                        format: 'essay',
                        label: 'Lab Report Assignment',
                        context: 'Answers should follow scientific/lab reporting style.',
                        weighttype: 'individual'
                    },
                    'custom-essay': {
                        format: 'essay',
                        label: 'Custom Assignment',
                        context: 'Follow teacher-defined custom assignment characteristics.',
                        weighttype: 'individual'
                    },
                    'multiple-choice': {
                        format: 'multiple-choice',
                        label: 'Quiz (Multiple Choice)',
                        context: 'Assessment should be objective and auto-checkable.',
                        weighttype: 'quiz'
                    }
                };

                const resolved = Object.prototype.hasOwnProperty.call(map, selected)
                    ? map[selected]
                    : map['essay'];
                if (selected !== 'custom-essay') {
                    return resolved;
                }

                const customlabel = assignTypeCustomInput
                    ? String(assignTypeCustomInput.value || '').trim()
                    : '';
                if (customlabel === '') {
                    return resolved;
                }

                return {
                    format: 'essay',
                    label: customlabel,
                    context: 'Follow teacher-defined custom assignment characteristics.',
                    weighttype: 'individual'
                };
            };

            const weightLabelMap = {
                'very-easy': {
                    label: 'Very Easy',
                    percent: 30
                },
                'easy': {
                    label: 'Easy',
                    percent: 50
                },
                'medium': {
                    label: 'Medium',
                    percent: 70
                },
                'hard': {
                    label: 'Hard',
                    percent: 85
                },
                'very-hard': {
                    label: 'Very Hard',
                    percent: 100
                }
            };

            const resolveWeightSelection = () => {
                const selected = assignWeightLabelInput
                    ? String(assignWeightLabelInput.value || 'medium').trim()
                    : 'medium';
                const fallback = weightLabelMap.medium;
                const resolved = Object.prototype.hasOwnProperty.call(weightLabelMap, selected)
                    ? weightLabelMap[selected]
                    : fallback;

                if (assignWeightPercentInput) {
                    assignWeightPercentInput.value = String(resolved.percent);
                }

                return {
                    key: Object.prototype.hasOwnProperty.call(weightLabelMap, selected) ? selected : 'medium',
                    label: resolved.label,
                    percent: resolved.percent
                };
            };

            const syncAssignmentTypeCustomState = () => {
                if (!assignTypeInput || !assignTypeCustomWrap || !assignTypeCustomInput) {
                    return;
                }
                const iscustom = String(assignTypeInput.value || '').trim() === 'custom-essay';
                assignTypeCustomWrap.style.display = iscustom ? 'block' : 'none';
                assignTypeCustomInput.disabled = !iscustom;
                if (!iscustom) {
                    assignTypeCustomInput.value = '';
                }
            };

            const isAssignmentCountConfigurable = () => {
                if (!assignTypeInput) {
                    return true;
                }
                const selected = String(assignTypeInput.value || '').trim().toLowerCase();
                return selected === 'essay' || selected === 'multiple-choice';
            };

            const syncAssignmentCountVisibility = () => {
                if (!assignCountWrap || !assignCountInput) {
                    return;
                }
                const enabled = isAssignmentCountConfigurable();
                assignCountWrap.style.display = enabled ? 'block' : 'none';
                assignCountInput.disabled = !enabled;
            };

            const buildAssignmentPrompt = () => {
                let className = '';
                let courseId = '';
                if (assignClassInput) {
                    courseId = String(assignClassInput.value || '').trim();
                    const selected = assignClassInput.options[assignClassInput.selectedIndex];
                    className = selected && selected.dataset && selected.dataset.coursename
                        ? String(selected.dataset.coursename).trim()
                        : String(selected ? selected.text : '').trim();
                }
                const topic = assignTopicInput ? String(assignTopicInput.value || '').trim() : '';
                const selectedPdf = assignPdfInput ? String(assignPdfInput.value || '').trim() : '';
                const assignmentSelection = resolveAssignmentSelection();
                const weightSelection = resolveWeightSelection();
                const countConfigurable = isAssignmentCountConfigurable();
                const count = countConfigurable ? normalizeQuestionCountInput(assignCountInput) : 0;
                const notes = assignNotesInput ? assignNotesInput.value.trim() : '';

                let taskFormatRule = '';
                if (assignmentSelection.format === 'multiple-choice') {
                    taskFormatRule = [
                        `Create exactly ${count} multiple-choice questions in English.`,
                        'Each question must include options A, B, C, and D.',
                        'Each question must have exactly ONE correct answer.',
                        'The other three options must be plausible distractors but still incorrect.',
                        'Avoid ambiguous options like "all answers are correct" or "all answers are wrong".',
                        'Question List format is mandatory:',
                        '1. <question text>',
                        'A) <option A>',
                        'B) <option B>',
                        'C) <option C>',
                        'D) <option D>',
                        'Repeat until all questions are listed.',
                        'Answer Key format is mandatory and concise:',
                        '1. A',
                        '2. C',
                        '3. B',
                        `Continue until ${count}.`,
                        'Do not include explanations in the answer key.'
                    ].join('\n');
                } else {
                    if (countConfigurable) {
                        taskFormatRule = [
                            `Create exactly ${count} essay questions/components in English.`,
                            'Each question must be clear, specific, and measurable.',
                            'Question List format is mandatory:',
                            '1. <question text>',
                            '2. <question text>',
                            `Continue until ${count}.`,
                            'Answer Key format is mandatory and concise:',
                            '1. <key points>',
                            '2. <key points>',
                            `Continue until ${count}.`
                        ].join('\n');
                    } else {
                        taskFormatRule = [
                            'Create essay questions/components in English based on teacher request and topic scope.',
                            'Do not force a fixed number unless explicitly requested in Additional Notes.',
                            'Each question must be clear, specific, and measurable.',
                            'Question List format is mandatory:',
                            '1. <question text>',
                            '2. <question text>',
                            'Continue sequentially until complete.',
                            'Answer Key format is mandatory and concise:',
                            '1. <key points>',
                            '2. <key points>',
                            'Provide matching entries for every generated question.'
                        ].join('\n');
                    }
                }

                return [
                    'You are a teaching assistant that generates a Moodle assignment draft.',
                    'Use clear, professional English only.',
                    `Class: ${className || courseId || '-'}`,
                    `Topic: ${topic || '-'}`,
                    `Reference Material (PDF): ${selectedPdf || '-'}`,
                    `Assignment Type: ${assignmentSelection.label}`,
                    `Task Context: ${assignmentSelection.context}`,
                    `Question Format: ${assignmentSelection.format === 'multiple-choice' ? 'Multiple Choice' : 'Essay'}`,
                    `Weight Label: ${weightSelection.label} (${weightSelection.percent}%)`,
                    `Number of Questions/Components: ${
                        countConfigurable ? String(count) : 'Not fixed (follow Additional Notes/topic scope)'
                    }`,
                    `Additional Notes: ${notes || '-'}`,
                    taskFormatRule,
                    'All content must be natural English and typo-free.',
                    'Do not use placeholders like [due date], [insert], or TBD.',
                    'You MUST follow this exact output structure (no extra sections):',
                    'Assignment Title:',
                    'Learning Objectives:',
                    'Instructions for Students:',
                    'Question List:',
                    'Answer Key:',
                    'Grading Rubric:',
                    'Restriction: do not include meta-openers like "## Answer" or "Here is".'
                ].join('\n');
            };

            const buildPracticePrompt = () => {
                let className = '';
                let courseId = '';
                if (practiceClassInput) {
                    courseId = String(practiceClassInput.value || '').trim();
                    const selected = practiceClassInput.options[practiceClassInput.selectedIndex];
                    className = selected && selected.dataset && selected.dataset.coursename
                        ? String(selected.dataset.coursename).trim()
                        : String(selected ? selected.text : '').trim();
                }
                const topic = practiceTopicInput ? String(practiceTopicInput.value || '').trim() : '';
                const selectedPdf = practicePdfInput ? String(practicePdfInput.value || '').trim() : '-';
                const count = normalizeQuestionCountInput(practiceCountInput);
                const pageRange = getPracticePageRange();
                const pageRangeText = (pageRange.pageStart > 0 && pageRange.pageEnd > 0)
                    ? `${pageRange.pageStart}-${pageRange.pageEnd}`
                    : 'all pages';

                return [
                    'You are a quiz question writer.',
                    'Output plain text only.',
                    `Class: ${className || courseId || '-'}`,
                    `Topic: ${topic || '-'}`,
                    `Reference Material (PDF): ${selectedPdf || '-'}`,
                    `Allowed pages: ${pageRangeText}. Use only this page range.`,
                    `Generate exactly ${count} multiple-choice questions in English.`,
                    'Each question must include 4 options: A), B), C), D).',
                    'Provide exactly one correct option per question.',
                    'Use this exact format only:',
                    'Question List:',
                    '1. <question>',
                    'A) <option>',
                    'B) <option>',
                    'C) <option>',
                    'D) <option>',
                    `Continue sequentially until ${count}.`,
                    'Answer Key:',
                    '1. A',
                    '2. B',
                    `Continue sequentially until ${count}.`,
                    'No explanations. No introductions. No extra sections.'
                ].join('\n');
            };

            const buildPracticeSingleQuestionPrompt = (questionNumber, totalCount) => {
                let className = '';
                let courseId = '';
                if (practiceClassInput) {
                    courseId = String(practiceClassInput.value || '').trim();
                    const selected = practiceClassInput.options[practiceClassInput.selectedIndex];
                    className = selected && selected.dataset && selected.dataset.coursename
                        ? String(selected.dataset.coursename).trim()
                        : String(selected ? selected.text : '').trim();
                }
                const topic = practiceTopicInput ? String(practiceTopicInput.value || '').trim() : '';
                const selectedPdf = practicePdfInput ? String(practicePdfInput.value || '').trim() : '-';
                const safeNumber = Number.isFinite(questionNumber) && questionNumber > 0 ? questionNumber : 1;
                const safeTotal = Number.isFinite(totalCount) && totalCount > 0 ? totalCount : 1;
                const pageRange = getPracticePageRange();
                const pageRangeText = (pageRange.pageStart > 0 && pageRange.pageEnd > 0)
                    ? `${pageRange.pageStart}-${pageRange.pageEnd}`
                    : 'all pages';

                return [
                    'You are a quiz question writer.',
                    'Output plain text only.',
                    `Class: ${className || courseId || '-'}`,
                    `Topic: ${topic || '-'}`,
                    `Reference Material (PDF): ${selectedPdf || '-'}`,
                    `Allowed pages: ${pageRangeText}. Use only this page range.`,
                    `Create question number ${safeNumber} out of ${safeTotal}.`,
                    'Generate exactly one multiple-choice question in English.',
                    'The question must include 4 options: A), B), C), D).',
                    'Provide exactly one correct option for this question.',
                    'Use this exact format only:',
                    'Question List:',
                    `${safeNumber}. <question>`,
                    'A) <option>',
                    'B) <option>',
                    'C) <option>',
                    'D) <option>',
                    'Answer Key:',
                    `${safeNumber}. <A/B/C/D>`,
                    'No explanations. No introductions. No extra sections.',
                    'Do NOT include analysis, self-talk, or reflective text.',
                    'Forbidden phrases include: "Wait", "However", "The user", "context provided", "I need to".',
                    'The question stem must be one clean exam question only.'
                ].join('\n');
            };

            const normalizeGeneratedDraft = (rawText) => {
                let text = String(rawText || '').trim();
                text = text.replace(/^```[a-zA-Z]*\s*/i, '').replace(/```$/i, '').trim();
                text = text.replace(/^##\s*Answer\s*/i, '').trim();
                text = text.replace(/^Answer\s*:?\s*/i, '').trim();
                text = text.replace(/^Berikut (adalah|ini).*?:\s*/i, '').trim();
                text = text.replace(/^Here (is|are).*?:\s*/i, '').trim();
                text = text.replace(/(\r?\n\s*){3,}/g, '\n\n').trim();
                return text;
            };

            const extractDraftSection = (text, startLabels, endLabels) => {
                const normalized = String(text || '').replace(/\r\n/g, '\n');
                if (!normalized.trim()) {
                    return '';
                }
                const lines = normalized.split('\n');
                let startIndex = -1;
                for (let i = 0; i < lines.length; i++) {
                    const line = String(lines[i] || '').trim().toLowerCase();
                    if (startLabels.some((label) => line === label || line.startsWith(`${label}:`))) {
                        startIndex = i;
                        break;
                    }
                }
                if (startIndex < 0) {
                    return '';
                }
                let endIndex = lines.length;
                for (let i = startIndex + 1; i < lines.length; i++) {
                    const line = String(lines[i] || '').trim().toLowerCase();
                    if (endLabels.some((label) => line === label || line.startsWith(`${label}:`))) {
                        endIndex = i;
                        break;
                    }
                }
                return lines.slice(startIndex + 1, endIndex).join('\n').trim();
            };

            const parsePracticeQuestionBank = (rawText) => {
                const normalized = normalizeGeneratedDraft(rawText);
                const titleBlock = extractDraftSection(
                    normalized,
                    ['assignment title', 'judul tugas', 'title'],
                    ['question list', 'questions', 'daftar soal', 'soal', 'answer key', 'kunci jawaban', 'correct answers', 'correct answer']
                );
                const questionBlock = extractDraftSection(
                    normalized,
                    ['question list', 'questions', 'daftar soal', 'soal'],
                    ['answer key', 'kunci jawaban', 'correct answers', 'correct answer', 'grading rubric', 'rubrik penilaian']
                );
                const answerBlock = extractDraftSection(
                    normalized,
                    ['answer key', 'kunci jawaban', 'correct answers', 'correct answer'],
                    ['grading rubric', 'rubrik penilaian']
                );

                const questionsSource = questionBlock || normalized;
                const answersSource = answerBlock || normalized;
                const questions = [];
                let current = null;
                let currentOption = '';
                const looksLikeQuestionLine = (line) => {
                    const text = String(line || '').trim();
                    if (!text) {
                        return false;
                    }
                    if (/\?$/.test(text)) {
                        return true;
                    }
                    return /^(what|why|how|which|when|where|who|apa|mengapa|bagaimana)\b/i.test(text);
                };
                const hasAllOptions = (question) => (
                    question &&
                    question.options &&
                    String(question.options.A || '').trim() &&
                    String(question.options.B || '').trim() &&
                    String(question.options.C || '').trim() &&
                    String(question.options.D || '').trim()
                );

                questionsSource.split('\n').forEach((lineRaw) => {
                    const line = String(lineRaw || '').trim();
                    if (!line) {
                        return;
                    }
                    const qMatch = line.match(/^(\d+)\s*[.)]\s*(.+)$/);
                    if (qMatch) {
                        if (current) {
                            questions.push(current);
                        }
                        current = {
                            sourceNumber: parseInt(qMatch[1], 10),
                            stem: String(qMatch[2] || '').trim(),
                            options: { A: '', B: '', C: '', D: '' }
                        };
                        currentOption = '';
                        return;
                    }
                    const optMatch = line.match(/^(?:[-*]\s*)?([A-Da-d])\s*[.)\:]\s*(.+)$/);
                    if (optMatch && current) {
                        currentOption = String(optMatch[1] || '').toUpperCase();
                        current.options[currentOption] = String(optMatch[2] || '').trim();
                        return;
                    }
                    if (current) {
                        if (looksLikeQuestionLine(line) && hasAllOptions(current)) {
                            questions.push(current);
                            current = {
                                sourceNumber: questions.length + 1,
                                stem: line,
                                options: { A: '', B: '', C: '', D: '' }
                            };
                            currentOption = '';
                            return;
                        }
                        if (currentOption && current.options[currentOption]) {
                            current.options[currentOption] = `${current.options[currentOption]} ${line}`.trim();
                        } else {
                            current.stem = `${current.stem} ${line}`.trim();
                        }
                    }
                });
                if (current) {
                    questions.push(current);
                }

                const keyByNumber = {};
                answersSource.split('\n').forEach((lineRaw) => {
                    const line = String(lineRaw || '').trim();
                    if (!line) {
                        return;
                    }
                    const keyMatch = line.match(/^(\d+)\s*[.)\-:]\s*(.+)$/);
                    if (!keyMatch) {
                        return;
                    }
                    const number = parseInt(keyMatch[1], 10);
                    const value = String(keyMatch[2] || '').toUpperCase();
                    const letterMatch = value.match(/[A-D]/);
                    if (Number.isFinite(number) && letterMatch) {
                        keyByNumber[number] = letterMatch[0];
                    }
                });

                const normalizedQuestions = questions
                    .filter((item) => item && item.stem)
                    .map((item, index) => {
                        const number = index + 1;
                        const sourceNumber = Number.isFinite(item.sourceNumber) ? item.sourceNumber : number;
                        const options = {
                            A: String(item.options.A || '').trim(),
                            B: String(item.options.B || '').trim(),
                            C: String(item.options.C || '').trim(),
                            D: String(item.options.D || '').trim()
                        };
                        return {
                            number,
                            sourceNumber,
                            stem: String(item.stem || '').trim(),
                            options,
                            answer: keyByNumber[sourceNumber] || keyByNumber[number] || 'A'
                        };
                    })
                    .filter((item) => item.stem && item.options.A && item.options.B && item.options.C && item.options.D);

                return {
                    title: String(titleBlock || '').split('\n').map((line) => line.trim()).filter(Boolean)[0] || '',
                    questions: normalizedQuestions
                };
            };

            const parseSinglePracticeQuestion = (rawText, questionNumber) => {
                const parsed = parsePracticeQuestionBank(rawText);
                if (!parsed || !Array.isArray(parsed.questions) || parsed.questions.length < 1) {
                    throw new Error(config.practiceparseerror || 'Failed to parse generated practice question.');
                }
                const first = parsed.questions[0];
                const stem = String(first.stem || '').trim();
                const hasMetaReasoning = /\b(wait|however|the user|context provided|i need to|let'?s think|based on the context|reference material for this question)\b/i.test(stem);
                if (!stem) {
                    throw new Error(config.practiceparseerror || 'Generated question stem is empty.');
                }
                if (stem.length > 320) {
                    throw new Error(config.practiceparseerror || 'Generated question stem is too long and likely invalid.');
                }
                if (hasMetaReasoning) {
                    throw new Error(config.practiceparseerror || 'Generated question contains meta reasoning text.');
                }
                const answer = String(first.answer || '').trim().toUpperCase();
                if (!['A', 'B', 'C', 'D'].includes(answer)) {
                    throw new Error(config.practiceparseerror || 'Failed to parse generated practice question answer key.');
                }
                return {
                    number: questionNumber,
                    sourceNumber: questionNumber,
                    stem,
                    options: {
                        A: String(first.options && first.options.A ? first.options.A : '').trim(),
                        B: String(first.options && first.options.B ? first.options.B : '').trim(),
                        C: String(first.options && first.options.C ? first.options.C : '').trim(),
                        D: String(first.options && first.options.D ? first.options.D : '').trim()
                    },
                    answer
                };
            };

            const buildPracticeProgressPreviewText = (statusLine) => {
                const total = Number.isFinite(practiceGenerationState.total) ? practiceGenerationState.total : 0;
                const generated = Array.isArray(practiceGenerationState.generatedQuestions)
                    ? practiceGenerationState.generatedQuestions.length
                    : 0;
                const listOnly = formatPracticeQuestionListOnly({
                    questions: practiceGenerationState.generatedQuestions
                });
                const parts = [
                    statusLine || `Generated ${generated}/${total} question(s).`
                ];
                if (generated > 0) {
                    parts.push('');
                    parts.push(listOnly);
                }
                return parts.join('\n').trim();
            };

            const buildPracticePublishTemplate = () => {
                const sourceText = String(practiceRawDraftText || practiceLastDraftText || '').trim();
                if (!sourceText) {
                    throw new Error(config.practicegeneratedfirst || 'Generate practice draft first before publishing.');
                }
                const parsed = parsePracticeQuestionBank(sourceText);
                if (!Array.isArray(parsed.questions) || parsed.questions.length < 1) {
                    throw new Error(config.practiceparseerror || 'Failed to prepare publish template from generated practice.');
                }

                const topic = practiceTopicInput ? String(practiceTopicInput.value || '').trim() : '';
                const fallbackTitle = topic ? `${topic} Practice Quiz` : 'Practice Quiz';
                const title = String(parsed.title || '').trim() || fallbackTitle;
                const total = parsed.questions.length;
                if (practiceCountInput) {
                    practiceCountInput.value = String(total);
                }

                const questionLines = [];
                const answerLines = [];
                parsed.questions.forEach((question) => {
                    questionLines.push(`${question.number}. ${question.stem}`);
                    questionLines.push(`A) ${question.options.A}`);
                    questionLines.push(`B) ${question.options.B}`);
                    questionLines.push(`C) ${question.options.C}`);
                    questionLines.push(`D) ${question.options.D}`);
                    answerLines.push(`${question.number}. ${question.answer}`);
                });

                return [
                    `Assignment Title: ${title}`,
                    '',
                    'Learning Objectives:',
                    `- Identify key concepts related to ${topic || 'the selected topic'}.`,
                    '- Apply understanding through multiple-choice reasoning.',
                    '- Evaluate closely related options and choose the best answer.',
                    '',
                    'Instructions for Students:',
                    '- Choose one best answer for each question.',
                    '- Read each question carefully before selecting an option.',
                    '- This practice is for self-learning and immediate feedback.',
                    '',
                    'Question List:',
                    ...questionLines,
                    '',
                    'Answer Key:',
                    ...answerLines,
                    '',
                    'Grading Rubric:',
                    '- 1 point per correct answer.',
                    `- Total score: ${total} points.`,
                    '- Suggested mastery target: 70% or above.'
                ].join('\n');
            };

            const formatPracticeQuestionBank = (parsed, fallbackTitle = 'Practice Quiz') => {
                const safeParsed = parsed && typeof parsed === 'object' ? parsed : {};
                const questions = Array.isArray(safeParsed.questions) ? safeParsed.questions : [];
                const title = String(safeParsed.title || '').trim() || fallbackTitle;
                const questionLines = [];
                const answerLines = [];
                questions.forEach((question, index) => {
                    const number = index + 1;
                    questionLines.push(`${number}. ${String(question.stem || '').trim()}`);
                    questionLines.push(`A) ${String(question.options && question.options.A ? question.options.A : '').trim()}`);
                    questionLines.push(`B) ${String(question.options && question.options.B ? question.options.B : '').trim()}`);
                    questionLines.push(`C) ${String(question.options && question.options.C ? question.options.C : '').trim()}`);
                    questionLines.push(`D) ${String(question.options && question.options.D ? question.options.D : '').trim()}`);
                    answerLines.push(`${number}. ${String(question.answer || 'A').trim()}`);
                });
                return [
                    `Assignment Title: ${title}`,
                    '',
                    'Question List:',
                    ...questionLines,
                    '',
                    'Answer Key:',
                    ...answerLines
                ].join('\n');
            };

            const formatPracticeQuestionListOnly = (parsed) => {
                const safeParsed = parsed && typeof parsed === 'object' ? parsed : {};
                const questions = Array.isArray(safeParsed.questions) ? safeParsed.questions : [];
                const questionLines = [];
                questions.forEach((question, index) => {
                    const number = index + 1;
                    questionLines.push(`${number}. ${String(question.stem || '').trim()}`);
                    questionLines.push(`A) ${String(question.options && question.options.A ? question.options.A : '').trim()}`);
                    questionLines.push(`B) ${String(question.options && question.options.B ? question.options.B : '').trim()}`);
                    questionLines.push(`C) ${String(question.options && question.options.C ? question.options.C : '').trim()}`);
                    questionLines.push(`D) ${String(question.options && question.options.D ? question.options.D : '').trim()}`);
                });
                return ['Question List:', ...questionLines].join('\n');
            };

            const appendPublishNote = (message, linkUrl, isError) => {
                if (!assignPreview) {
                    return;
                }
                const hr = document.createElement('hr');
                const note = document.createElement('p');
                note.className = `local-chatbot-publish-note${isError ? ' is-error' : ''}`;
                note.textContent = message || '';
                assignPreview.appendChild(hr);
                assignPreview.appendChild(note);
                if (linkUrl) {
                    const link = document.createElement('a');
                    link.href = linkUrl;
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    link.textContent = linkUrl;
                    assignPreview.appendChild(link);
                }
            };

            const appendPracticePublishNote = (message, linkUrl, isError) => {
                if (!practicePreview) {
                    return;
                }
                const hr = document.createElement('hr');
                const note = document.createElement('p');
                note.className = `local-chatbot-publish-note${isError ? ' is-error' : ''}`;
                note.textContent = message || '';
                practicePreview.appendChild(hr);
                practicePreview.appendChild(note);
                if (linkUrl) {
                    const link = document.createElement('a');
                    link.href = linkUrl;
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    link.textContent = linkUrl;
                    practicePreview.appendChild(link);
                }
            };

            const setAssignmentTopicOptions = (topics, placeholderText) => {
                if (!assignTopicInput) {
                    return;
                }
                assignTopicInput.innerHTML = '';
                if (placeholderText) {
                    const placeholder = document.createElement('option');
                    placeholder.value = '';
                    placeholder.textContent = placeholderText;
                    assignTopicInput.appendChild(placeholder);
                }
                if (!Array.isArray(topics)) {
                    return;
                }
                topics.forEach((topic) => {
                    const option = document.createElement('option');
                    option.value = String(topic.value || topic.label || '');
                    option.textContent = String(topic.label || topic.value || '');
                    assignTopicInput.appendChild(option);
                });
                if (assignTopicInput.options.length > 1) {
                    assignTopicInput.selectedIndex = 1;
                } else {
                    assignTopicInput.selectedIndex = 0;
                }
            };

            const loadAssignmentTopics = async () => {
                if (!assignClassInput || !assignTopicInput) {
                    return;
                }
                const courseid = String(assignClassInput.value || '').trim();
                if (!courseid) {
                    setAssignmentTopicOptions([], config.assignmenttopicplaceholder || 'Select class first');
                    return;
                }

                if (Object.prototype.hasOwnProperty.call(configuredCourseTopics, courseid)) {
                    const localTopics = Array.isArray(configuredCourseTopics[courseid]) ? configuredCourseTopics[courseid] : [];
                    if (localTopics.length > 0) {
                        setAssignmentTopicOptions(localTopics, config.assignmenttopicplaceholder || 'Select a topic');
                    } else {
                        setAssignmentTopicOptions([], config.assignmenttopicempty || 'No topics found in this class');
                    }
                    return;
                }
                setAssignmentTopicOptions([], config.assignmenttopicloading || 'Loading topics...');
                const form = new FormData();
                form.append('action', 'course_topics');
                form.append('sesskey', config.sesskey);
                form.append('courseid', courseid);
                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload.ok) {
                        throw new Error(payload.error || config.assignmenttopicempty || 'No topics found');
                    }
                    const topics = Array.isArray(payload.topics) ? payload.topics : [];
                    if (!topics.length) {
                        setAssignmentTopicOptions([], config.assignmenttopicempty || 'No topics found in this class');
                        return;
                    }
                    setAssignmentTopicOptions(topics, config.assignmenttopicplaceholder || 'Select a topic');
                } catch (err) {
                    setAssignmentTopicOptions([], err.message || config.assignmenttopicempty || 'No topics found in this class');
                }
            };

            const setAssignmentPdfOptions = (pdfs, placeholderText) => {
                if (!assignPdfInput) {
                    return;
                }
                assignPdfInput.innerHTML = '';
                if (placeholderText) {
                    const placeholder = document.createElement('option');
                    placeholder.value = '';
                    placeholder.textContent = placeholderText;
                    assignPdfInput.appendChild(placeholder);
                }
                if (!Array.isArray(pdfs)) {
                    return;
                }
                pdfs.forEach((pdf) => {
                    const option = document.createElement('option');
                    option.value = String(pdf.value || pdf.label || '');
                    option.textContent = String(pdf.label || pdf.value || '');
                    assignPdfInput.appendChild(option);
                });
                if (assignPdfInput.options.length > 1) {
                    assignPdfInput.selectedIndex = 1;
                } else {
                    assignPdfInput.selectedIndex = 0;
                }
            };

            const loadAssignmentPdfs = async () => {
                if (!assignClassInput || !assignPdfInput) {
                    return;
                }
                const courseidRaw = String(assignClassInput.value || '').trim();
                const selectedClassOption = assignClassInput.options[assignClassInput.selectedIndex];
                const courseName = selectedClassOption && selectedClassOption.dataset && selectedClassOption.dataset.coursename
                    ? String(selectedClassOption.dataset.coursename).trim()
                    : String(selectedClassOption ? selectedClassOption.text : '').trim();
                if (!courseidRaw && !courseName) {
                    setAssignmentPdfOptions([], 'Pilih kelas dulu');
                    setTopicMasteryBadge(assignMasteryBadge, taskMasteryUnknown, 'neutral');
                    return;
                }
                const topic = assignTopicInput ? String(assignTopicInput.value || '').trim() : '';
                if (!topic) {
                    setAssignmentPdfOptions([], config.assignmentpdfplaceholder || 'Pilih topik dulu');
                    setTopicMasteryBadge(assignMasteryBadge, taskMasteryUnknown, 'neutral');
                    return;
                }
                await updateAssignmentMastery(courseidRaw, courseName, topic);

                const localPdfList = Object.prototype.hasOwnProperty.call(configuredCoursePdfs, courseidRaw)
                    ? configuredCoursePdfs[courseidRaw]
                    : (Object.prototype.hasOwnProperty.call(configuredCoursePdfs, courseName) ? configuredCoursePdfs[courseName] : null);
                if (Array.isArray(localPdfList)) {
                    const normalizedTopic = topic.toLowerCase().trim();
                    const filteredLocal = localPdfList.filter((pdf) => {
                        const itemTopic = String(pdf.topic || '').toLowerCase().trim();
                        return itemTopic === normalizedTopic;
                    });
                    if (!filteredLocal.length) {
                        setAssignmentPdfOptions([], config.assignmentpdfempty || 'No PDF resource found in this class');
                    } else {
                        setAssignmentPdfOptions(filteredLocal, config.assignmentpdfplaceholder || 'Pilih PDF');
                    }
                    return;
                }

                setAssignmentPdfOptions([], config.assignmentpdfloading || 'Loading PDFs...');
                const form = new FormData();
                form.append('action', 'course_pdfs');
                form.append('sesskey', config.sesskey);
                const numericCourseId = /^\d+$/.test(courseidRaw) ? courseidRaw : '0';
                form.append('courseid', numericCourseId);
                form.append('course_name', courseName);
                form.append('topic', topic);
                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload.ok) {
                        throw new Error(payload.error || config.assignmentpdfempty || 'No PDF found');
                    }
                    const pdfs = Array.isArray(payload.pdfs) ? payload.pdfs : [];
                    if (!pdfs.length) {
                        setAssignmentPdfOptions([], config.assignmentpdfempty || 'No PDF resource found in this class');
                        return;
                    }
                    setAssignmentPdfOptions(pdfs, config.assignmentpdfplaceholder || 'Select a PDF');
                } catch (err) {
                    setAssignmentPdfOptions([], err.message || config.assignmentpdfempty || 'No PDF resource found in this class');
                }
            };

            const setPracticeTopicOptions = (topics, placeholderText) => {
                if (!practiceTopicInput || practiceTopicInput.tagName !== 'SELECT') {
                    return;
                }
                practiceTopicInput.innerHTML = '';
                if (placeholderText) {
                    const placeholder = document.createElement('option');
                    placeholder.value = '';
                    placeholder.textContent = placeholderText;
                    practiceTopicInput.appendChild(placeholder);
                }
                if (!Array.isArray(topics)) {
                    return;
                }
                topics.forEach((topic) => {
                    const option = document.createElement('option');
                    option.value = String(topic.value || topic.label || '');
                    option.textContent = String(topic.label || topic.value || '');
                    practiceTopicInput.appendChild(option);
                });
                if (practiceTopicInput.options.length > 1) {
                    practiceTopicInput.selectedIndex = 1;
                } else {
                    practiceTopicInput.selectedIndex = 0;
                }
            };

            const loadPracticeTopics = async () => {
                if (!practiceClassInput || !practiceTopicInput || practiceTopicInput.tagName !== 'SELECT') {
                    return;
                }
                const courseid = String(practiceClassInput.value || '').trim();
                if (!courseid) {
                    setPracticeTopicOptions([], config.assignmenttopicplaceholder || 'Select class first');
                    return;
                }

                if (Object.prototype.hasOwnProperty.call(configuredCourseTopics, courseid)) {
                    const localTopics = Array.isArray(configuredCourseTopics[courseid]) ? configuredCourseTopics[courseid] : [];
                    if (localTopics.length > 0) {
                        setPracticeTopicOptions(localTopics, config.assignmenttopicplaceholder || 'Select a topic');
                    } else {
                        setPracticeTopicOptions([], config.assignmenttopicempty || 'No topics found in this class');
                    }
                    return;
                }

                setPracticeTopicOptions([], config.assignmenttopicloading || 'Loading topics...');
                const form = new FormData();
                form.append('action', 'course_topics');
                form.append('sesskey', config.sesskey);
                form.append('courseid', courseid);
                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload.ok) {
                        throw new Error(payload.error || config.assignmenttopicempty || 'No topics found');
                    }
                    const topics = Array.isArray(payload.topics) ? payload.topics : [];
                    if (!topics.length) {
                        setPracticeTopicOptions([], config.assignmenttopicempty || 'No topics found in this class');
                        return;
                    }
                    setPracticeTopicOptions(topics, config.assignmenttopicplaceholder || 'Select a topic');
                } catch (err) {
                    setPracticeTopicOptions([], err.message || config.assignmenttopicempty || 'No topics found in this class');
                }
            };

            const setPracticePdfOptions = (pdfs, placeholderText) => {
                if (!practicePdfInput) {
                    return;
                }
                practicePdfInput.innerHTML = '';
                if (placeholderText) {
                    const placeholder = document.createElement('option');
                    placeholder.value = '';
                    placeholder.textContent = placeholderText;
                    practicePdfInput.appendChild(placeholder);
                }
                if (!Array.isArray(pdfs)) {
                    return;
                }
                pdfs.forEach((pdf) => {
                    const option = document.createElement('option');
                    option.value = String(pdf.value || pdf.label || '');
                    option.textContent = String(pdf.label || pdf.value || '');
                    practicePdfInput.appendChild(option);
                });
                if (practicePdfInput.options.length > 1) {
                    practicePdfInput.selectedIndex = 1;
                } else {
                    practicePdfInput.selectedIndex = 0;
                }
            };

            const loadPracticePdfs = async () => {
                if (!practiceClassInput || !practiceTopicInput || !practicePdfInput) {
                    return;
                }
                const courseidRaw = String(practiceClassInput.value || '').trim();
                const selectedClassOption = practiceClassInput.options[practiceClassInput.selectedIndex];
                const courseName = selectedClassOption && selectedClassOption.dataset && selectedClassOption.dataset.coursename
                    ? String(selectedClassOption.dataset.coursename).trim()
                    : String(selectedClassOption ? selectedClassOption.text : '').trim();
                if (!courseidRaw && !courseName) {
                    setPracticePdfOptions([], 'Pilih kelas dulu');
                    setTopicMasteryBadge(practiceMasteryBadge, taskMasteryUnknown, 'neutral');
                    return;
                }
                const topic = String(practiceTopicInput.value || '').trim();
                if (!topic) {
                    setPracticePdfOptions([], config.assignmentpdfplaceholder || 'Pilih topik dulu');
                    setTopicMasteryBadge(practiceMasteryBadge, taskMasteryUnknown, 'neutral');
                    return;
                }
                await updatePracticeMastery(courseidRaw, courseName, topic);

                const localPdfList = Object.prototype.hasOwnProperty.call(configuredCoursePdfs, courseidRaw)
                    ? configuredCoursePdfs[courseidRaw]
                    : (Object.prototype.hasOwnProperty.call(configuredCoursePdfs, courseName) ? configuredCoursePdfs[courseName] : null);
                if (Array.isArray(localPdfList)) {
                    const normalizedTopic = topic.toLowerCase().trim();
                    const filteredLocal = localPdfList.filter((pdf) => {
                        const itemTopic = String(pdf.topic || '').toLowerCase().trim();
                        return itemTopic === normalizedTopic;
                    });
                    if (!filteredLocal.length) {
                        setPracticePdfOptions([], config.assignmentpdfempty || 'No PDF resource found in this class');
                    } else {
                        setPracticePdfOptions(filteredLocal, config.assignmentpdfplaceholder || 'Pilih PDF');
                    }
                    return;
                }

                setPracticePdfOptions([], config.assignmentpdfloading || 'Loading PDFs...');
                const form = new FormData();
                form.append('action', 'course_pdfs');
                form.append('sesskey', config.sesskey);
                const numericCourseId = /^\d+$/.test(courseidRaw) ? courseidRaw : '0';
                form.append('courseid', numericCourseId);
                form.append('course_name', courseName);
                form.append('topic', topic);
                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload.ok) {
                        throw new Error(payload.error || config.assignmentpdfempty || 'No PDF found');
                    }
                    const pdfs = Array.isArray(payload.pdfs) ? payload.pdfs : [];
                    if (!pdfs.length) {
                        setPracticePdfOptions([], config.assignmentpdfempty || 'No PDF resource found in this class');
                        return;
                    }
                    setPracticePdfOptions(pdfs, config.assignmentpdfplaceholder || 'Select a PDF');
                } catch (err) {
                    setPracticePdfOptions([], err.message || config.assignmentpdfempty || 'No PDF resource found in this class');
                }
            };

            const generateAssignmentDraft = async () => {
                if (!assignPreview) {
                    return;
                }
                if (!config.isteacher) {
                    await setGeneratedContent(assignPreview, config.roleteacheronly || 'Teacher only feature.', false);
                    return;
                }
                await setGeneratedContent(assignPreview, `${config.assignmentgenerate || 'Generate Draft'}...`, false);
                if (assignGenerateBtn) {
                    assignGenerateBtn.disabled = true;
                }
                if (assignRegenerateBtn) {
                    assignRegenerateBtn.disabled = true;
                }
                try {
                    const classValue = assignClassInput ? String(assignClassInput.value || '').trim() : '';
                    const topicValue = assignTopicInput ? String(assignTopicInput.value || '').trim() : '';
                    const pdfValue = assignPdfInput ? String(assignPdfInput.value || '').trim() : '';
                    if (!classValue) {
                        throw new Error('Pilih kelas tujuan terlebih dahulu.');
                    }
                    if (!topicValue) {
                        throw new Error('Pilih topik terlebih dahulu.');
                    }
                    if (!pdfValue) {
                        throw new Error('Pilih materi (PDF) terlebih dahulu.');
                    }

                    const selected = assignClassInput ? assignClassInput.options[assignClassInput.selectedIndex] : null;
                    const className = selected && selected.dataset && selected.dataset.coursename
                        ? String(selected.dataset.coursename).trim()
                        : String(selected ? selected.text : '').trim();
                    const materialForm = new FormData();
                    materialForm.append('action', 'set_material_context');
                    materialForm.append('sesskey', config.sesskey);
                    materialForm.append('courseid', classValue);
                    materialForm.append('course_name', className);
                    materialForm.append('topic', topicValue);
                    const materialPayload = await postForm(config.ajaxurl, materialForm);
                    if (!materialPayload || !materialPayload.ok) {
                        throw new Error(
                            (materialPayload && materialPayload.error)
                                ? materialPayload.error
                                : (config.chaterror || 'Failed to load materials.')
                        );
                    }

                    assignmentLastPrompt = buildAssignmentPrompt();
                    const payload = await runChatRequest(assignmentLastPrompt, {
                        generationMode: 'assignment'
                    });
                    assignmentLastDraftText = normalizeGeneratedDraft(payload.answer || '');
                    await setGeneratedContent(assignPreview, assignmentLastDraftText, true);
                } catch (err) {
                    assignmentLastDraftText = '';
                    await setGeneratedContent(assignPreview, err.message || config.chaterror, false);
                } finally {
                    if (assignGenerateBtn) {
                        assignGenerateBtn.disabled = false;
                    }
                    if (assignRegenerateBtn) {
                        assignRegenerateBtn.disabled = false;
                    }
                }
            };

            const regenerateAssignmentDraft = async () => {
                if (!assignPreview) {
                    return;
                }
                await setGeneratedContent(assignPreview, `${config.assignmentregenerate || 'Regenerate'}...`, false);
                if (assignGenerateBtn) {
                    assignGenerateBtn.disabled = true;
                }
                if (assignRegenerateBtn) {
                    assignRegenerateBtn.disabled = true;
                }
                try {
                    const classValue = assignClassInput ? String(assignClassInput.value || '').trim() : '';
                    const topicValue = assignTopicInput ? String(assignTopicInput.value || '').trim() : '';
                    const pdfValue = assignPdfInput ? String(assignPdfInput.value || '').trim() : '';
                    if (!classValue) {
                        throw new Error('Pilih kelas tujuan terlebih dahulu.');
                    }
                    if (!topicValue) {
                        throw new Error('Pilih topik terlebih dahulu.');
                    }
                    if (!pdfValue) {
                        throw new Error('Pilih materi (PDF) terlebih dahulu.');
                    }

                    const selected = assignClassInput ? assignClassInput.options[assignClassInput.selectedIndex] : null;
                    const className = selected && selected.dataset && selected.dataset.coursename
                        ? String(selected.dataset.coursename).trim()
                        : String(selected ? selected.text : '').trim();
                    const materialForm = new FormData();
                    materialForm.append('action', 'set_material_context');
                    materialForm.append('sesskey', config.sesskey);
                    materialForm.append('courseid', classValue);
                    materialForm.append('course_name', className);
                    materialForm.append('topic', topicValue);
                    const materialPayload = await postForm(config.ajaxurl, materialForm);
                    if (!materialPayload || !materialPayload.ok) {
                        throw new Error(
                            (materialPayload && materialPayload.error)
                                ? materialPayload.error
                                : (config.chaterror || 'Failed to load materials.')
                        );
                    }

                    assignmentLastPrompt = buildAssignmentPrompt();
                    const payload = await runChatRequest(assignmentLastPrompt, {
                        generationMode: 'assignment'
                    });
                    assignmentLastDraftText = normalizeGeneratedDraft(payload.answer || '');
                    await setGeneratedContent(assignPreview, assignmentLastDraftText, true);
                } catch (err) {
                    assignmentLastDraftText = '';
                    await setGeneratedContent(assignPreview, err.message || config.chaterror, false);
                } finally {
                    if (assignGenerateBtn) {
                        assignGenerateBtn.disabled = false;
                    }
                    if (assignRegenerateBtn) {
                        assignRegenerateBtn.disabled = false;
                    }
                }
            };

            const publishAssignmentDraft = async () => {
                if (!assignPreview) {
                    return;
                }
                const courseid = assignClassInput ? String(assignClassInput.value || '').trim() : '';
                const topic = assignTopicInput ? String(assignTopicInput.value || '').trim() : '';
                if (!courseid) {
                    appendPublishNote(
                        config.assignmentselectclassfirst || 'Select target class first.',
                        '',
                        true
                    );
                    return;
                }

                const text = String(assignmentLastDraftText || assignPreview.textContent || '').trim();
                if (!text || text === config.assignmentplaceholder) {
                    appendPublishNote(
                        config.assignmentgeneratedfirst || 'Generate draft first before publishing.',
                        '',
                        true
                    );
                    return;
                }

                if (!config.savedrafturl || !config.publishurl) {
                    appendPublishNote('Publish endpoint is not configured.', '', true);
                    return;
                }

                const previousLabel = assignPublishBtn ? assignPublishBtn.textContent : '';
                if (assignPublishBtn) {
                    assignPublishBtn.disabled = true;
                    assignPublishBtn.textContent = config.assignmentpublishing || 'Publishing...';
                }
                if (assignGenerateBtn) {
                    assignGenerateBtn.disabled = true;
                }
                if (assignRegenerateBtn) {
                    assignRegenerateBtn.disabled = true;
                }

                try {
                    const normalizedCount = isAssignmentCountConfigurable()
                        ? normalizeQuestionCountInput(assignCountInput)
                        : 0;
                    const assignmentSelection = resolveAssignmentSelection();
                    const weightSelection = resolveWeightSelection();
                    const saveForm = new FormData();
                    saveForm.append('sesskey', config.sesskey);
                    saveForm.append('courseid', courseid);
                    saveForm.append('topic', topic);
                    saveForm.append('content_mode', 'assignment');
                    saveForm.append('assignment_type', assignmentSelection.format);
                    saveForm.append('assignment_type_label', assignmentSelection.label);
                    saveForm.append('weight_bucket_type', assignmentSelection.weighttype);
                    saveForm.append('activity_weight_label', weightSelection.key);
                    saveForm.append('activity_weight_percent', String(weightSelection.percent));
                    saveForm.append('weight_source', 'llm');
                    saveForm.append('question_count', normalizedCount);
                    saveForm.append('essay_autograde_enabled', '0');
                    saveForm.append('draft_text', text);

                    const savePayload = await postForm(config.savedrafturl, saveForm);
                    if (!savePayload || !savePayload.success || !savePayload.draftid) {
                        throw new Error(
                            (savePayload && savePayload.message)
                                ? savePayload.message
                                : (config.assignmentpublisherror || 'Failed to publish draft.')
                        );
                    }

                    const publishForm = new FormData();
                    publishForm.append('sesskey', config.sesskey);
                    publishForm.append('courseid', courseid);
                    publishForm.append('draftid', String(savePayload.draftid));

                    const publishPayload = await postForm(config.publishurl, publishForm);
                    if (!publishPayload || !publishPayload.success) {
                        throw new Error(
                            (publishPayload && publishPayload.message)
                                ? publishPayload.message
                                : (config.assignmentpublisherror || 'Failed to publish draft.')
                        );
                    }

                    appendPublishNote(
                        publishPayload.message || config.assignmentpublished || 'Published.',
                        publishPayload.url || '',
                        false
                    );
                } catch (err) {
                    appendPublishNote(
                        err && err.message ? err.message : (config.assignmentpublisherror || 'Failed to publish draft.'),
                        '',
                        true
                    );
                } finally {
                    if (assignPublishBtn) {
                        assignPublishBtn.disabled = false;
                        assignPublishBtn.textContent = previousLabel || (config.assignmentpublish || 'Publish');
                    }
                    if (assignGenerateBtn) {
                        assignGenerateBtn.disabled = false;
                    }
                    if (assignRegenerateBtn) {
                        assignRegenerateBtn.disabled = false;
                    }
                }
            };

            const resolvePracticeGenerationContext = async () => {
                let courseid = '';
                let topic = '';
                const pageRange = getPracticePageRange();
                if (practiceClassInput) {
                    courseid = String(practiceClassInput.value || '').trim();
                    topic = practiceTopicInput ? String(practiceTopicInput.value || '').trim() : '';
                    const selectedPdf = practicePdfInput ? String(practicePdfInput.value || '').trim() : '';
                    const selected = practiceClassInput.options[practiceClassInput.selectedIndex];
                    const className = selected && selected.dataset && selected.dataset.coursename
                        ? String(selected.dataset.coursename).trim()
                        : String(selected ? selected.text : '').trim();

                    if (!courseid) {
                        throw new Error('Pilih kelas terlebih dahulu.');
                    }
                    if (!topic) {
                        throw new Error('Pilih topik terlebih dahulu.');
                    }
                    if (practicePdfInput && !selectedPdf) {
                        throw new Error('Pilih materi (PDF) terlebih dahulu.');
                    }

                    const materialForm = new FormData();
                    materialForm.append('action', 'set_material_context');
                    materialForm.append('sesskey', config.sesskey);
                    materialForm.append('courseid', courseid);
                    materialForm.append('course_name', className);
                    materialForm.append('topic', topic);
                    if (pageRange.pageStart > 0) {
                        materialForm.append('page_start', String(pageRange.pageStart));
                    }
                    if (pageRange.pageEnd > 0) {
                        materialForm.append('page_end', String(pageRange.pageEnd));
                    }
                    const materialPayload = await postForm(config.ajaxurl, materialForm);
                    if (!materialPayload || !materialPayload.ok) {
                        throw new Error(
                            (materialPayload && materialPayload.error)
                                ? materialPayload.error
                                : (config.chaterror || 'Failed to load materials.')
                        );
                    }
                } else {
                    topic = practiceTopicInput ? String(practiceTopicInput.value || '').trim() : '';
                    if (!topic) {
                        throw new Error('Isi topik practice terlebih dahulu.');
                    }
                }
                return {
                    courseid,
                    topic,
                    pageStart: pageRange.pageStart,
                    pageEnd: pageRange.pageEnd
                };
            };

            const syncPracticeDraftFromGeneratedQuestions = () => {
                const topic = practiceTopicInput ? String(practiceTopicInput.value || '').trim() : '';
                const fallbackTitle = topic ? `${topic} Practice Quiz` : 'Practice Quiz';
                const generatedQuestions = Array.isArray(practiceGenerationState.generatedQuestions)
                    ? practiceGenerationState.generatedQuestions
                    : [];
                if (!generatedQuestions.length) {
                    practiceRawDraftText = '';
                    practiceLastDraftText = '';
                    return;
                }
                const parsed = {
                    title: fallbackTitle,
                    questions: generatedQuestions.map((question, index) => ({
                        number: index + 1,
                        sourceNumber: index + 1,
                        stem: String(question.stem || '').trim(),
                        options: {
                            A: String(question.options && question.options.A ? question.options.A : '').trim(),
                            B: String(question.options && question.options.B ? question.options.B : '').trim(),
                            C: String(question.options && question.options.C ? question.options.C : '').trim(),
                            D: String(question.options && question.options.D ? question.options.D : '').trim()
                        },
                        answer: String(question.answer || 'A').trim().toUpperCase()
                    }))
                };
                practiceRawDraftText = formatPracticeQuestionBank(parsed, fallbackTitle);
                practiceLastDraftText = formatPracticeQuestionListOnly(parsed);
            };

            const generateSinglePracticeQuestionWithRetry = async (questionNumber, totalCount, chatOptions = {}) => {
                const maxAttempts = 3;
                let lastError = null;
                for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                    if (practiceStopRequested) {
                        throw new Error('Practice generation stopped by user.');
                    }
                    const statusLine = `Generated ${practiceGenerationState.generatedQuestions.length}/${totalCount} question(s). Generating question ${questionNumber}/${totalCount} (attempt ${attempt}/${maxAttempts})...`;
                    await setGeneratedContent(practicePreview, buildPracticeProgressPreviewText(statusLine), false);
                    try {
                        const prompt = buildPracticeSingleQuestionPrompt(questionNumber, totalCount);
                        const requestId = generateRequestId();
                        practiceActiveAbortController = new AbortController();
                        const payload = await runChatRequest(prompt, {
                            ...chatOptions,
                            generationMode: 'practice',
                            requestId,
                            questionNumber,
                            attempt,
                            signal: practiceActiveAbortController.signal
                        });
                        const generatedText = normalizeGeneratedDraft(payload.answer || '');
                        return parseSinglePracticeQuestion(generatedText, questionNumber);
                    } catch (err) {
                        if (practiceStopRequested || isAbortError(err) || isPracticeStopError(err)) {
                            throw new Error('Practice generation stopped by user.');
                        }
                        lastError = err;
                    } finally {
                        practiceActiveAbortController = null;
                    }
                }
                const errorMessage = lastError && lastError.message
                    ? lastError.message
                    : (config.chaterror || 'Failed to process chat request.');
                throw new Error(`Question ${questionNumber}/${totalCount} failed after 3 retries. ${errorMessage}`);
            };

            const runPracticeGenerationLoop = async (startQuestionNumber, totalCount, chatOptions = {}) => {
                for (let number = startQuestionNumber; number <= totalCount; number++) {
                    if (practiceStopRequested) {
                        throw new Error('Practice generation stopped by user.');
                    }
                    const question = await generateSinglePracticeQuestionWithRetry(number, totalCount, chatOptions);
                    const normalizedQuestion = {
                        number,
                        sourceNumber: number,
                        stem: String(question.stem || '').trim(),
                        options: {
                            A: String(question.options && question.options.A ? question.options.A : '').trim(),
                            B: String(question.options && question.options.B ? question.options.B : '').trim(),
                            C: String(question.options && question.options.C ? question.options.C : '').trim(),
                            D: String(question.options && question.options.D ? question.options.D : '').trim()
                        },
                        answer: String(question.answer || 'A').trim().toUpperCase()
                    };
                    const existingIndex = practiceGenerationState.generatedQuestions.findIndex(
                        (item) => Number(item.number) === number
                    );
                    if (existingIndex >= 0) {
                        practiceGenerationState.generatedQuestions[existingIndex] = normalizedQuestion;
                    } else {
                        practiceGenerationState.generatedQuestions.push(normalizedQuestion);
                    }
                    practiceGenerationState.generatedQuestions.sort(
                        (a, b) => Number(a.number || 0) - Number(b.number || 0)
                    );
                    syncPracticeDraftFromGeneratedQuestions();
                    const successLine = `Generated ${practiceGenerationState.generatedQuestions.length}/${totalCount} question(s).`;
                    await setGeneratedContent(practicePreview, buildPracticeProgressPreviewText(successLine), false);
                }
            };

            const continuePracticeGeneration = async (startQuestionNumber, totalCount, chatOptions = {}) => {
                practiceStopRequested = false;
                practiceGenerationInProgress = true;
                practiceGenerationState.blocked = false;
                practiceGenerationState.failedQuestionNumber = 0;
                practiceGenerationState.lastErrorMessage = '';
                setPracticeRecoveryActionsVisible(false);
                setPracticeGenerationControlsDisabled(true);
                setPracticeStopButtonState(true);

                try {
                    await runPracticeGenerationLoop(startQuestionNumber, totalCount, chatOptions);
                    practiceGenerationState.blocked = false;
                    practiceGenerationState.failedQuestionNumber = 0;
                    syncPracticeDraftFromGeneratedQuestions();
                    await setGeneratedContent(
                        practicePreview,
                        buildPracticeProgressPreviewText(
                            `Generated ${practiceGenerationState.generatedQuestions.length}/${totalCount} question(s).`
                        ),
                        false
                    );
                    appendPracticePublishNote(
                        config.practicephase1hint || 'Question bank generated. Click Prepare publish template before publishing.',
                        '',
                        false
                    );
                } catch (err) {
                    if (practiceStopRequested || isPracticeStopError(err) || isAbortError(err)) {
                        practiceGenerationState.blocked = false;
                        practiceGenerationState.failedQuestionNumber = 0;
                        practiceGenerationState.lastErrorMessage = 'Stopped by user.';
                        syncPracticeDraftFromGeneratedQuestions();
                        const stoppedLine = `Generated ${practiceGenerationState.generatedQuestions.length}/${totalCount} question(s). Generation stopped by user.`;
                        await setGeneratedContent(practicePreview, buildPracticeProgressPreviewText(stoppedLine), false);
                        appendPracticePublishNote(
                            'Generation stopped. You can continue with generated questions or run Generate again.',
                            '',
                            false
                        );
                        return;
                    }
                    practiceGenerationState.blocked = true;
                    practiceGenerationState.failedQuestionNumber = startQuestionNumber
                        + practiceGenerationState.generatedQuestions.filter(
                            (item) => Number(item.number || 0) >= startQuestionNumber
                        ).length;
                    practiceGenerationState.lastErrorMessage = err && err.message
                        ? err.message
                        : (config.chaterror || 'Failed to process chat request.');
                    syncPracticeDraftFromGeneratedQuestions();
                    const blockedLine = `Generated ${practiceGenerationState.generatedQuestions.length}/${totalCount} question(s). Generation stopped at question ${practiceGenerationState.failedQuestionNumber}.`;
                    const details = [
                        blockedLine,
                        `Error: ${practiceGenerationState.lastErrorMessage}`,
                        'Choose: Retry failed question, or Continue to publish phase with generated questions.'
                    ].join('\n');
                    await setGeneratedContent(practicePreview, buildPracticeProgressPreviewText(details), false);
                    setPracticeRecoveryActionsVisible(true);
                    appendPracticePublishNote(
                        'Generation stopped after max retries. Retry the failed question or continue to publish with current output.',
                        '',
                        true
                    );
                } finally {
                    practiceGenerationInProgress = false;
                    practiceStopRequested = false;
                    practiceActiveAbortController = null;
                    setPracticeGenerationControlsDisabled(false);
                    setPracticeStopButtonState(false);
                }
            };

            const generatePracticeDraft = async () => {
                if (!practicePreview) {
                    return;
                }
                resetPracticePreparedState();
                resetPracticeGenerationState();
                practiceLastPrompt = buildPracticePrompt();
                const totalCount = normalizeQuestionCountInput(practiceCountInput);
                practiceGenerationState.total = totalCount;
                practiceGenerationState.generatedQuestions = [];
                setPracticeGenerationControlsDisabled(true);

                try {
                    const chatOptions = await resolvePracticeGenerationContext();
                    await setGeneratedContent(
                        practicePreview,
                        buildPracticeProgressPreviewText(`Generated 0/${totalCount} question(s). Starting generation...`),
                        false
                    );
                    await continuePracticeGeneration(1, totalCount, chatOptions);
                } catch (err) {
                    practiceRawDraftText = '';
                    practicePreparedDraftText = '';
                    practiceIsPreparedForPublish = false;
                    practiceLastDraftText = '';
                    await setGeneratedContent(
                        practicePreview,
                        err && err.message ? err.message : (config.chaterror || 'Failed to process chat request.'),
                        false
                    );
                    setPracticeGenerationControlsDisabled(false);
                }
            };

            const stopPracticeGeneration = async () => {
                if (!practiceGenerationInProgress) {
                    return;
                }
                practiceStopRequested = true;
                if (practiceStopBtn) {
                    practiceStopBtn.disabled = true;
                }
                if (practiceActiveAbortController) {
                    practiceActiveAbortController.abort();
                }
            };

            const preparePracticeDraftForPublish = async () => {
                if (!practicePreview) {
                    return;
                }
                if (practicePrepareBtn) {
                    practicePrepareBtn.disabled = true;
                }
                if (practiceGenerateBtn) {
                    practiceGenerateBtn.disabled = true;
                }
                if (practicePublishBtn) {
                    practicePublishBtn.disabled = true;
                }
                try {
                    const prepared = buildPracticePublishTemplate();
                    practicePreparedDraftText = normalizeGeneratedDraft(prepared);
                    practiceLastDraftText = practicePreparedDraftText;
                    practiceIsPreparedForPublish = true;
                    await setGeneratedContent(practicePreview, practicePreparedDraftText, true);
                    appendPracticePublishNote(
                        config.practiceprepared || 'Publish template is ready. Review and click Publish practice.',
                        '',
                        false
                    );
                } catch (err) {
                    practiceIsPreparedForPublish = false;
                    appendPracticePublishNote(
                        err && err.message ? err.message : (config.practiceparseerror || 'Failed to prepare publish template from generated practice.'),
                        '',
                        true
                    );
                } finally {
                    if (practicePrepareBtn) {
                        practicePrepareBtn.disabled = false;
                    }
                    if (practiceGenerateBtn) {
                        practiceGenerateBtn.disabled = false;
                    }
                    if (practicePublishBtn) {
                        practicePublishBtn.disabled = false;
                    }
                }
            };

            const publishPracticeDraft = async () => {
                if (!practicePreview) {
                    return;
                }
                if (!practiceClassInput || !practiceTopicInput || !practicePdfInput) {
                    appendPracticePublishNote(
                        config.practicepublisherror || 'Failed to publish practice draft.',
                        '',
                        true
                    );
                    return;
                }

                const courseid = String(practiceClassInput.value || '').trim();
                const topic = String(practiceTopicInput.value || '').trim();
                if (!courseid) {
                    appendPracticePublishNote(
                        config.assignmentselectclassfirst || 'Select target class first.',
                        '',
                        true
                    );
                    return;
                }

                if (!practiceIsPreparedForPublish) {
                    appendPracticePublishNote(
                        config.practicepreparefirst || 'Prepare publish template first before publishing.',
                        '',
                        true
                    );
                    return;
                }

                const text = String(practicePreparedDraftText || practiceLastDraftText || practicePreview.textContent || '').trim();
                if (!text || text === config.practiceplaceholder) {
                    appendPracticePublishNote(
                        config.practicegeneratedfirst || 'Generate practice draft first before publishing.',
                        '',
                        true
                    );
                    return;
                }

                if (!config.savedrafturl || !config.publishurl) {
                    appendPracticePublishNote(config.practicepublisherror || 'Failed to publish practice draft.', '', true);
                    return;
                }

                const previousGenerateLabel = practiceGenerateBtn ? practiceGenerateBtn.textContent : '';
                const previousPrepareLabel = practicePrepareBtn ? practicePrepareBtn.textContent : '';
                const previousPublishLabel = practicePublishBtn ? practicePublishBtn.textContent : '';
                if (practicePublishBtn) {
                    practicePublishBtn.disabled = true;
                    practicePublishBtn.textContent = config.practicepublishing || 'Publishing practice...';
                }
                if (practiceGenerateBtn) {
                    practiceGenerateBtn.disabled = true;
                }
                if (practicePrepareBtn) {
                    practicePrepareBtn.disabled = true;
                }

                try {
                    const normalizedCount = normalizeQuestionCountInput(practiceCountInput);
                    const saveForm = new FormData();
                    saveForm.append('sesskey', config.sesskey);
                    saveForm.append('courseid', courseid);
                    saveForm.append('topic', topic);
                    saveForm.append('content_mode', 'practice');
                    saveForm.append('assignment_type', 'multiple-choice');
                    saveForm.append('question_count', normalizedCount);
                    saveForm.append('draft_text', text);

                    const savePayload = await postForm(config.savedrafturl, saveForm);
                    if (!savePayload || !savePayload.success || !savePayload.draftid) {
                        throw new Error(
                            (savePayload && savePayload.message)
                                ? savePayload.message
                                : (config.practicepublisherror || 'Failed to publish practice draft.')
                        );
                    }

                    const publishForm = new FormData();
                    publishForm.append('sesskey', config.sesskey);
                    publishForm.append('courseid', courseid);
                    publishForm.append('draftid', String(savePayload.draftid));

                    const publishPayload = await postForm(config.publishurl, publishForm);
                    if (!publishPayload || !publishPayload.success) {
                        throw new Error(
                            (publishPayload && publishPayload.message)
                                ? publishPayload.message
                                : (config.practicepublisherror || 'Failed to publish practice draft.')
                        );
                    }

                    appendPracticePublishNote(
                        publishPayload.message || config.practicepublished || 'Practice published.',
                        publishPayload.url || '',
                        false
                    );
                } catch (err) {
                    appendPracticePublishNote(
                        err && err.message ? err.message : (config.practicepublisherror || 'Failed to publish practice draft.'),
                        '',
                        true
                    );
                } finally {
                    if (practiceGenerateBtn) {
                        practiceGenerateBtn.disabled = false;
                        if (previousGenerateLabel) {
                            practiceGenerateBtn.textContent = previousGenerateLabel;
                        }
                    }
                    if (practicePrepareBtn) {
                        practicePrepareBtn.disabled = false;
                        if (previousPrepareLabel) {
                            practicePrepareBtn.textContent = previousPrepareLabel;
                        }
                    }
                    if (practicePublishBtn) {
                        practicePublishBtn.disabled = false;
                        practicePublishBtn.textContent = previousPublishLabel || (config.practicepublish || 'Publish practice');
                    }
                }
            };

            const retryFailedPracticeQuestion = async () => {
                if (!practicePreview) {
                    return;
                }
                const totalCount = Number.isFinite(practiceGenerationState.total) && practiceGenerationState.total > 0
                    ? practiceGenerationState.total
                    : normalizeQuestionCountInput(practiceCountInput);
                const failedNumber = Number.isFinite(practiceGenerationState.failedQuestionNumber) &&
                    practiceGenerationState.failedQuestionNumber > 0
                    ? practiceGenerationState.failedQuestionNumber
                    : (practiceGenerationState.generatedQuestions.length + 1);

                resetPracticePreparedState();
                try {
                    const chatOptions = await resolvePracticeGenerationContext();
                    await continuePracticeGeneration(failedNumber, totalCount, chatOptions);
                } catch (err) {
                    await setGeneratedContent(
                        practicePreview,
                        err && err.message ? err.message : (config.chaterror || 'Failed to process chat request.'),
                        false
                    );
                    setPracticeGenerationControlsDisabled(false);
                }
            };

            const continueToPracticePublishPhase = async () => {
                if (!practicePreview) {
                    return;
                }
                resetPracticePreparedState();
                syncPracticeDraftFromGeneratedQuestions();
                const generatedCount = Array.isArray(practiceGenerationState.generatedQuestions)
                    ? practiceGenerationState.generatedQuestions.length
                    : 0;
                if (generatedCount < 1) {
                    await setGeneratedContent(
                        practicePreview,
                        config.practicegeneratedfirst || 'Generate practice draft first before publishing.',
                        false
                    );
                    return;
                }
                setPracticeRecoveryActionsVisible(false);
                await setGeneratedContent(
                    practicePreview,
                    buildPracticeProgressPreviewText(
                        `Generated ${generatedCount}/${practiceGenerationState.total} question(s). Continue with Prepare publish template or Publish practice.`
                    ),
                    false
                );
                appendPracticePublishNote(
                    'You can continue to prepare/publish with generated questions, or run Retry failed question later by generating again.',
                    '',
                    false
                );
            };

            resetPracticeGenerationState();
            restoreHistory();
            setMasteryBadge(config.chatmasteryunknown || 'Mastery: select class and topic', 'neutral');
            setTopicMasteryBadge(assignMasteryBadge, taskMasteryUnknown, 'neutral');
            setTopicMasteryBadge(practiceMasteryBadge, taskMasteryUnknown, 'neutral');
            if (chatClassInput) {
                const handleChatClassChange = async () => {
                    await loadChatTopics();
                    await syncChatMaterials();
                };
                chatClassInput.addEventListener('change', handleChatClassChange);
                handleChatClassChange();
            } else {
                syncChatMaterials();
            }
            if (chatTopicInput) {
                chatTopicInput.addEventListener('change', syncChatMaterials);
            }
            switchTab('chat');

            if (sendBtn) {
                sendBtn.addEventListener('click', sendMessage);
            }
            if (clearBtn) {
                clearBtn.addEventListener('click', clearHistory);
            }
            tabButtons.forEach((button) => {
                button.addEventListener('click', () => {
                    const tabName = button.dataset.tab || 'chat';
                    switchTab(tabName);
                });
            });
            if (assignGenerateBtn) {
                assignGenerateBtn.addEventListener('click', generateAssignmentDraft);
            }
            if (assignRegenerateBtn) {
                assignRegenerateBtn.addEventListener('click', regenerateAssignmentDraft);
            }
            if (assignPublishBtn) {
                assignPublishBtn.addEventListener('click', publishAssignmentDraft);
            }
            if (practiceGenerateBtn) {
                practiceGenerateBtn.addEventListener('click', generatePracticeDraft);
            }
            if (practiceStopBtn) {
                practiceStopBtn.addEventListener('click', stopPracticeGeneration);
            }
            if (practiceRetryQuestionBtn) {
                practiceRetryQuestionBtn.addEventListener('click', retryFailedPracticeQuestion);
            }
            if (practiceContinuePublishBtn) {
                practiceContinuePublishBtn.addEventListener('click', continueToPracticePublishPhase);
            }
            if (practicePrepareBtn) {
                practicePrepareBtn.addEventListener('click', preparePracticeDraftForPublish);
            }
            if (practicePublishBtn) {
                practicePublishBtn.addEventListener('click', publishPracticeDraft);
            }
            if (assignClassInput) {
                const handleClassChange = async () => {
                    await loadAssignmentTopics();
                    await loadAssignmentPdfs();
                };
                assignClassInput.addEventListener('change', handleClassChange);
                handleClassChange();
            }
            if (assignTopicInput) {
                assignTopicInput.addEventListener('change', loadAssignmentPdfs);
            }
            if (assignTypeInput) {
                assignTypeInput.addEventListener('change', () => {
                    syncAssignmentTypeCustomState();
                    syncAssignmentCountVisibility();
                });
                syncAssignmentTypeCustomState();
                syncAssignmentCountVisibility();
            }
            if (assignWeightLabelInput) {
                assignWeightLabelInput.addEventListener('change', resolveWeightSelection);
                resolveWeightSelection();
            }
            if (practiceClassInput && practiceTopicInput && practiceTopicInput.tagName === 'SELECT') {
                const handlePracticeClassChange = async () => {
                    resetPracticePreparedState();
                    resetPracticeGenerationState();
                    await loadPracticeTopics();
                    await loadPracticePdfs();
                };
                practiceClassInput.addEventListener('change', handlePracticeClassChange);
                handlePracticeClassChange();
            }
            if (practiceTopicInput && practicePdfInput && practiceTopicInput.tagName === 'SELECT') {
                practiceTopicInput.addEventListener('change', async () => {
                    resetPracticePreparedState();
                    resetPracticeGenerationState();
                    await loadPracticePdfs();
                });
            }
            if (assignCountInput) {
                assignCountInput.addEventListener('change', () => normalizeQuestionCountInput(assignCountInput));
                assignCountInput.addEventListener('blur', () => normalizeQuestionCountInput(assignCountInput));
            }
            if (practiceCountInput) {
                practiceCountInput.addEventListener('change', () => {
                    normalizeQuestionCountInput(practiceCountInput);
                    resetPracticePreparedState();
                    resetPracticeGenerationState();
                });
                practiceCountInput.addEventListener('blur', () => {
                    normalizeQuestionCountInput(practiceCountInput);
                    resetPracticePreparedState();
                    resetPracticeGenerationState();
                });
            }
            if (practicePageStartInput) {
                const onPracticePageRangeChanged = () => {
                    getPracticePageRange();
                    resetPracticePreparedState();
                    resetPracticeGenerationState();
                };
                practicePageStartInput.addEventListener('change', onPracticePageRangeChanged);
                practicePageStartInput.addEventListener('blur', onPracticePageRangeChanged);
            }
            if (practicePageEndInput) {
                const onPracticePageRangeChanged = () => {
                    getPracticePageRange();
                    resetPracticePreparedState();
                    resetPracticeGenerationState();
                };
                practicePageEndInput.addEventListener('change', onPracticePageRangeChanged);
                practicePageEndInput.addEventListener('blur', onPracticePageRangeChanged);
            }
            if (input) {
                input.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
            }
        }
    };
});
