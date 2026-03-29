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

    const appendMessageDom = (text, type, sources) => {
        const messages = document.getElementById('local-chatbot-messages');
        if (!messages) {
            return null;
        }

        const item = document.createElement('div');
        item.className = `local-chatbot-message ${type}`;
        item.textContent = text;
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

    const postForm = async (url, data) => {
        const res = await fetch(url, {
            method: 'POST',
            body: data,
            credentials: 'same-origin',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        });
        return res.json();
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

            const assignClassInput = document.getElementById('local-chatbot-assign-class');
            const assignTopicInput = document.getElementById('local-chatbot-assign-topic');
            const assignPdfInput = document.getElementById('local-chatbot-assign-pdf');
            const assignTypeInput = document.getElementById('local-chatbot-assign-type');
            const assignCountInput = document.getElementById('local-chatbot-assign-count');
            const assignNotesInput = document.getElementById('local-chatbot-assign-notes');
            const assignGenerateBtn = document.getElementById('local-chatbot-assign-generate');
            const assignRegenerateBtn = document.getElementById('local-chatbot-assign-regenerate');
            const assignPublishBtn = document.getElementById('local-chatbot-assign-publish');
            const assignPreview = document.getElementById('local-chatbot-assign-preview');

            const practiceClassInput = document.getElementById('local-chatbot-practice-class');
            const practiceTopicInput = document.getElementById('local-chatbot-practice-topic');
            const practicePdfInput = document.getElementById('local-chatbot-practice-pdf');
            const practiceCountInput = document.getElementById('local-chatbot-practice-count');
            const practiceGenerateBtn = document.getElementById('local-chatbot-practice-generate');
            const practicePublishBtn = document.getElementById('local-chatbot-practice-publish');
            const practicePreview = document.getElementById('local-chatbot-practice-preview');

            const storageKey = `local_chatbot_history_u${config.userid || 'anon'}`;
            const configuredCourseTopics = (config && typeof config.coursetopics === 'object' && config.coursetopics !== null)
                ? config.coursetopics
                : {};
            const configuredCoursePdfs = (config && typeof config.coursepdfs === 'object' && config.coursepdfs !== null)
                ? config.coursepdfs
                : {};

            let history = safeReadHistory(storageKey);
            let selectedFile = null;
            let assignmentLastPrompt = '';
            let assignmentLastDraftText = '';
            let practiceLastPrompt = '';
            let practiceLastDraftText = '';

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

            const appendMessage = (text, type, sources, persist) => {
                appendMessageDom(text, type, sources);
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

            const runChatRequest = async (question) => {
                const form = new FormData();
                form.append('action', 'chat');
                form.append('sesskey', config.sesskey);
                form.append('question', question);
                const payload = await postForm(config.ajaxurl, form);
                if (!payload.ok) {
                    throw new Error(payload.error || config.chaterror);
                }
                return payload;
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
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload.ok) {
                        throw new Error(payload.error || config.nofiles || 'No materials found');
                    }
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
                    const payload = await runChatRequest(question);
                    if (pending) {
                        pending.remove();
                    }
                    appendMessage(payload.answer || '', 'bot', payload.sources || [], true);
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
                const assignmentType = assignTypeInput ? String(assignTypeInput.value || 'essay').trim() : 'essay';
                const assignmentTypeLabel = assignmentType === 'multiple-choice'
                    ? 'Multiple Choice'
                    : 'Essay';
                const count = normalizeQuestionCountInput(assignCountInput);
                const notes = assignNotesInput ? assignNotesInput.value.trim() : '';

                let taskFormatRule = '';
                if (assignmentType === 'multiple-choice') {
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
                }

                return [
                    'You are a teaching assistant that generates a Moodle assignment draft.',
                    'Use clear, professional English only.',
                    `Class: ${className || courseId || '-'}`,
                    `Topic: ${topic || '-'}`,
                    `Reference Material (PDF): ${selectedPdf || '-'}`,
                    `Assignment Type: ${assignmentTypeLabel}`,
                    `Number of Questions/Components: ${count}`,
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

                const taskFormatRule = [
                    `Create exactly ${count} multiple-choice practice questions in English.`,
                    'Each question must include options A, B, C, and D.',
                    'Each question must have exactly ONE correct answer.',
                    'The other three options must be plausible distractors.',
                    'Question List format is mandatory:',
                    '1. <question text>',
                    'A) <option A>',
                    'B) <option B>',
                    'C) <option C>',
                    'D) <option D>',
                    `Continue until ${count}.`,
                    'Answer Key format is mandatory and concise:',
                    '1. A',
                    '2. C',
                    '3. B',
                    `Continue until ${count}.`,
                    'Do not include explanations inside Answer Key.'
                ].join('\n');

                return [
                    'You are a teaching assistant that generates a Moodle practice quiz draft.',
                    'Use clear, professional English only.',
                    `Class: ${className || courseId || '-'}`,
                    `Topic: ${topic || '-'}`,
                    `Reference Material (PDF): ${selectedPdf || '-'}`,
                    'Assignment Type: Multiple Choice',
                    `Number of Questions/Components: ${count}`,
                    'Additional Notes: Practice mode, designed for self-learning and immediate feedback.',
                    taskFormatRule,
                    'All content must be natural English and typo-free.',
                    'Do not use placeholders like [due date], [insert], or TBD.',
                    'You MUST follow this exact output structure (no extra sections):',
                    'Assignment Title:',
                    'Learning Objectives:',
                    'Instructions for Students:',
                    'Question List:',
                    'Answer Key:',
                    'Grading Rubric:'
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
                    return;
                }
                const topic = assignTopicInput ? String(assignTopicInput.value || '').trim() : '';
                if (!topic) {
                    setAssignmentPdfOptions([], config.assignmentpdfplaceholder || 'Pilih topik dulu');
                    return;
                }

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
                    return;
                }
                const topic = String(practiceTopicInput.value || '').trim();
                if (!topic) {
                    setPracticePdfOptions([], config.assignmentpdfplaceholder || 'Pilih topik dulu');
                    return;
                }

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
                const classValue = assignClassInput ? String(assignClassInput.value || '').trim() : '';
                const topicValue = assignTopicInput ? String(assignTopicInput.value || '').trim() : '';
                const pdfValue = assignPdfInput ? String(assignPdfInput.value || '').trim() : '';
                if (!classValue) {
                    await setGeneratedContent(assignPreview, 'Pilih kelas tujuan terlebih dahulu.', false);
                    return;
                }
                if (!topicValue) {
                    await setGeneratedContent(assignPreview, 'Pilih topik terlebih dahulu.', false);
                    return;
                }
                if (!pdfValue) {
                    await setGeneratedContent(assignPreview, 'Pilih materi (PDF) terlebih dahulu.', false);
                    return;
                }
                assignmentLastPrompt = buildAssignmentPrompt();
                await setGeneratedContent(assignPreview, `${config.assignmentgenerate || 'Generate Draft'}...`, false);
                if (assignGenerateBtn) {
                    assignGenerateBtn.disabled = true;
                }
                if (assignRegenerateBtn) {
                    assignRegenerateBtn.disabled = true;
                }
                try {
                    const payload = await runChatRequest(assignmentLastPrompt);
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
                if (!assignmentLastPrompt) {
                    assignmentLastPrompt = buildAssignmentPrompt();
                }
                await setGeneratedContent(assignPreview, `${config.assignmentregenerate || 'Regenerate'}...`, false);
                if (assignGenerateBtn) {
                    assignGenerateBtn.disabled = true;
                }
                if (assignRegenerateBtn) {
                    assignRegenerateBtn.disabled = true;
                }
                try {
                    const payload = await runChatRequest(assignmentLastPrompt);
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
                    const normalizedCount = normalizeQuestionCountInput(assignCountInput);
                    const saveForm = new FormData();
                    saveForm.append('sesskey', config.sesskey);
                    saveForm.append('courseid', courseid);
                    saveForm.append('topic', topic);
                    saveForm.append('content_mode', 'assignment');
                    saveForm.append('assignment_type', assignTypeInput ? String(assignTypeInput.value || 'essay') : 'essay');
                    saveForm.append('question_count', normalizedCount);
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

            const generatePracticeDraft = async () => {
                if (!practicePreview) {
                    return;
                }

                if (practiceClassInput) {
                    const courseid = String(practiceClassInput.value || '').trim();
                    const topic = practiceTopicInput ? String(practiceTopicInput.value || '').trim() : '';
                    const selectedPdf = practicePdfInput ? String(practicePdfInput.value || '').trim() : '';
                    const selected = practiceClassInput.options[practiceClassInput.selectedIndex];
                    const className = selected && selected.dataset && selected.dataset.coursename
                        ? String(selected.dataset.coursename).trim()
                        : String(selected ? selected.text : '').trim();

                    if (!courseid) {
                        await setGeneratedContent(practicePreview, 'Pilih kelas terlebih dahulu.', false);
                        return;
                    }
                    if (!topic) {
                        await setGeneratedContent(practicePreview, 'Pilih topik terlebih dahulu.', false);
                        return;
                    }
                    if (practicePdfInput && !selectedPdf) {
                        await setGeneratedContent(practicePreview, 'Pilih materi (PDF) terlebih dahulu.', false);
                        return;
                    }

                    const materialForm = new FormData();
                    materialForm.append('action', 'set_material_context');
                    materialForm.append('sesskey', config.sesskey);
                    materialForm.append('courseid', courseid);
                    materialForm.append('course_name', className);
                    materialForm.append('topic', topic);
                    const materialPayload = await postForm(config.ajaxurl, materialForm);
                    if (!materialPayload || !materialPayload.ok) {
                        await setGeneratedContent(
                            practicePreview,
                            (materialPayload && materialPayload.error)
                                ? materialPayload.error
                                : (config.chaterror || 'Failed to load materials.'),
                            false
                        );
                        return;
                    }
                } else {
                    const topic = practiceTopicInput ? String(practiceTopicInput.value || '').trim() : '';
                    if (!topic) {
                        await setGeneratedContent(practicePreview, 'Isi topik practice terlebih dahulu.', false);
                        return;
                    }
                }

                practiceLastPrompt = buildPracticePrompt();

                await setGeneratedContent(practicePreview, `${config.practicegenerate || 'Generate Practice'}...`, false);
                if (practiceGenerateBtn) {
                    practiceGenerateBtn.disabled = true;
                }
                if (practicePublishBtn) {
                    practicePublishBtn.disabled = true;
                }
                try {
                    const payload = await runChatRequest(practiceLastPrompt);
                    practiceLastDraftText = normalizeGeneratedDraft(payload.answer || '');
                    await setGeneratedContent(practicePreview, practiceLastDraftText, true);
                } catch (err) {
                    practiceLastDraftText = '';
                    await setGeneratedContent(practicePreview, err.message || config.chaterror, false);
                } finally {
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

                const text = String(practiceLastDraftText || practicePreview.textContent || '').trim();
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
                const previousPublishLabel = practicePublishBtn ? practicePublishBtn.textContent : '';
                if (practicePublishBtn) {
                    practicePublishBtn.disabled = true;
                    practicePublishBtn.textContent = config.practicepublishing || 'Publishing practice...';
                }
                if (practiceGenerateBtn) {
                    practiceGenerateBtn.disabled = true;
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
                    if (practicePublishBtn) {
                        practicePublishBtn.disabled = false;
                        practicePublishBtn.textContent = previousPublishLabel || (config.practicepublish || 'Publish practice');
                    }
                }
            };

            restoreHistory();
            if (config.isteacher) {
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
            } else {
                renderFiles([], config.nofiles, loadPreview, null);
                setPreviewEmpty(config.previewempty || 'Click a file to preview');
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
            if (practiceClassInput && practiceTopicInput && practiceTopicInput.tagName === 'SELECT') {
                const handlePracticeClassChange = async () => {
                    await loadPracticeTopics();
                    await loadPracticePdfs();
                };
                practiceClassInput.addEventListener('change', handlePracticeClassChange);
                handlePracticeClassChange();
            }
            if (practiceTopicInput && practicePdfInput && practiceTopicInput.tagName === 'SELECT') {
                practiceTopicInput.addEventListener('change', loadPracticePdfs);
            }
            if (assignCountInput) {
                assignCountInput.addEventListener('change', () => normalizeQuestionCountInput(assignCountInput));
                assignCountInput.addEventListener('blur', () => normalizeQuestionCountInput(assignCountInput));
            }
            if (practiceCountInput) {
                practiceCountInput.addEventListener('change', () => normalizeQuestionCountInput(practiceCountInput));
                practiceCountInput.addEventListener('blur', () => normalizeQuestionCountInput(practiceCountInput));
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
