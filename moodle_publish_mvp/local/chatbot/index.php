<?php
require_once(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/locallib.php');
require_once($CFG->dirroot . '/enrol/locallib.php');

require_login();

$context = context_system::instance();
require_capability('local/chatbot:view', $context);

$PAGE->set_context($context);
$PAGE->set_url(new moodle_url('/local/chatbot/index.php'));
$PAGE->set_pagelayout('standard');
$PAGE->add_body_class('local-chatbot-fullwidth');
$PAGE->set_title(get_string('pluginname', 'local_chatbot'));
$PAGE->set_heading(get_string('pluginname', 'local_chatbot'));

$chatcourses = [];
$coursetopicsmap = [];
$canmanualupload = local_chatbot_user_is_teacher_like((int)$USER->id) || is_siteadmin();
$canrefreshembedding = has_capability('local/chatbot:view', $context);

$courses = enrol_get_users_courses(
    (int)$USER->id,
    true,
    'id,shortname,fullname',
    'sortorder ASC'
);

foreach ($courses as $course) {
    if ((int)$course->id === SITEID) {
        continue;
    }
    if (!local_chatbot_user_can_access_course_materials((int)$course->id, (int)$USER->id)) {
        continue;
    }

    $chatcourses[] = $course;
    $topics = local_chatbot_list_course_topics((int)$course->id, (int)$USER->id);
    $courselabel = trim((string)$course->fullname) !== ''
        ? $course->fullname
        : $course->shortname;
    $coursetopicsmap[(string)$course->id] = $topics;
    $coursetopicsmap[$courselabel] = $topics;
}

$frontendbootconfig = local_chatbot_build_frontend_boot_config(
    (int)$USER->id,
    sesskey(),
    $coursetopicsmap,
    $canmanualupload,
    $canrefreshembedding
);
$frontendbootconfigjson = json_encode($frontendbootconfig, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
if ($frontendbootconfigjson === false) {
    $frontendbootconfigjson = '{}';
}
$embeddingconfig = local_chatbot_get_embedding_runtime_config();
$defaultchatmode = $frontendbootconfig['defaultchatmode'];
$embeddingconfigtitle = $frontendbootconfig['embeddingconfigtitle'];
$embeddingconfigactive = $frontendbootconfig['embeddingconfigactive'];
$embeddingconfigbackend = $frontendbootconfig['embeddingconfigbackend'];
$embeddingconfigollama = $frontendbootconfig['embeddingconfigollama'];
$embeddingconfigbert = $frontendbootconfig['embeddingconfigbert'];
$activeembeddingtext = $frontendbootconfig['activeembeddingtext'];
$refreshembeddingbutton = $frontendbootconfig['refreshembeddingbutton'];
$classplaceholder = $frontendbootconfig['courseclassplaceholder'];
$topicplaceholder = $frontendbootconfig['coursetopicplaceholder'];

$PAGE->requires->css('/local/chatbot/styles.css');
$PAGE->requires->js_init_code('window.localChatbotBootConfig = ' . $frontendbootconfigjson . ';');
$PAGE->requires->js_call_amd('local_chatbot/app', 'init', [$frontendbootconfig]);
$PAGE->requires->js_call_amd('local_chatbot/widget', 'init', [$frontendbootconfig]);
$PAGE->requires->js_init_code('(function() {
    const cfg = window.localChatbotBootConfig || null;
    const boot = function() {
        if (cfg && cfg.appownsmaterialspreview) {
            return;
        }
        const previewName = document.getElementById("local-chatbot-preview-name");
        const previewStatus = document.getElementById("local-chatbot-preview-embedding-status");
        const refreshBtn = document.getElementById("local-chatbot-refresh-embedding-btn");
        if (!previewName || !previewStatus || !refreshBtn || !cfg || !cfg.ajaxurl || !cfg.sesskey) {
            return;
        }

        let lastFilename = "";

        const escapeHtml = (value) => String(value || "").replace(/[&<>"\']/g, (ch) => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            "\"": "&quot;",
            "\'": "&#39;"
        }[ch] || ch));

        const getActiveModel = (backend) => {
            const normalized = String(backend || cfg.embedbackenddefault || "").trim().toLowerCase();
            if (normalized === "bert") {
                return String(cfg.embedmodelbert || "-");
            }
            if (normalized === "ollama") {
                return String(cfg.embedmodelollama || "-");
            }
            if (String(cfg.embedmodelbert || "").trim() !== "") {
                return String(cfg.embedmodelbert || "-");
            }
            return String(cfg.embedmodelollama || "-");
        };

        const humanizeStatus = (value) => {
            const normalized = String(value || "").trim().toLowerCase();
            if (normalized === "parsed") {
                return "parsed";
            }
            if (normalized === "needs_parsing") {
                return "needs parsing";
            }
            if (normalized === "no_materials") {
                return "no materials";
            }
            return normalized || "-";
        };

        const formatUnixTime = (value) => {
            const numeric = Math.max(0, parseInt(value, 10) || 0);
            if (!numeric) {
                return "-";
            }
            try {
                return new Date(numeric * 1000).toLocaleString();
            } catch (error) {
                return "-";
            }
        };

        const renderPlaceholder = () => {
            previewStatus.textContent = "Select a file to view embedding status.";
        };

        const renderStatus = (filename, info) => {
            const fileInActiveCorpus = Boolean(info && info.file_in_active_corpus);
            const isEmbedded = Boolean(info && info.is_embedded_in_active_index);
            const parseStatus = humanizeStatus(info && info.parse_status);
            const parsedAt = formatUnixTime(info && info.parsed_at);
            const embeddingBackend = String((info && info.embedding_backend) || cfg.embedbackenddefault || "").trim().toLowerCase();
            const embeddingModel = String((info && info.embedding_model) || getActiveModel(embeddingBackend));
            previewStatus.innerHTML =
                "<div><strong>Selected file:</strong> " + escapeHtml(filename) + "</div>"
                + "<div><strong>Index scope:</strong> active corpus</div>"
                + "<div><strong>File in active corpus:</strong> " + (fileInActiveCorpus ? "yes" : "no") + "</div>"
                + "<div><strong>Embedded in current index:</strong> " + (isEmbedded ? "yes" : "no") + "</div>"
                + "<div><strong>Index status:</strong> " + escapeHtml(parseStatus) + "</div>"
                + "<div><strong>Current embedding model:</strong> " + escapeHtml(embeddingModel) + "</div>"
                + "<div><strong>Last indexed:</strong> " + escapeHtml(parsedAt) + "</div>";
        };

        const postForm = async (form) => {
            const response = await fetch(cfg.ajaxurl, {
                method: "POST",
                body: form,
                credentials: "same-origin",
                headers: {"X-Requested-With": "XMLHttpRequest"}
            });
            const text = await response.text();
            return text && text.trim() ? JSON.parse(text) : {};
        };

        const getActiveFilename = () => {
            const value = String(previewName.textContent || "").trim();
            return value !== "" && value !== "-" ? value : "";
        };

        const syncButtonState = () => {
            if (!refreshBtn.disabled) {
                return;
            }
            refreshBtn.disabled = getActiveFilename() === "";
        };

        const fetchEmbeddingStatus = async () => {
            const filename = getActiveFilename();
            syncButtonState();
            if (!filename) {
                lastFilename = "";
                renderPlaceholder();
                return;
            }
            if (filename === lastFilename && previewStatus.textContent.indexOf("Select a file") === -1) {
                return;
            }
            lastFilename = filename;
            const form = new FormData();
            form.append("action", "file_content");
            form.append("sesskey", cfg.sesskey);
            form.append("filename", filename);
            try {
                const payload = await postForm(form);
                renderStatus(filename, payload.embedding_status || {});
            } catch (error) {
                renderStatus(filename, {});
            }
        };

        if (typeof MutationObserver !== "undefined") {
            const observer = new MutationObserver(() => {
                fetchEmbeddingStatus();
            });
            observer.observe(previewName, {childList: true, characterData: true, subtree: true});
        }

        syncButtonState();
        fetchEmbeddingStatus();
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot, {once: true});
    } else {
        boot();
    }
})();');

echo $OUTPUT->header();
?>
<div
    id="<?php echo s($frontendbootconfig['approotid']); ?>"
    class="local-chatbot-app-root"
    data-renderer-mode="<?php echo s($frontendbootconfig['renderermode']); ?>"
    data-boot-version="<?php echo (int)$frontendbootconfig['bootversion']; ?>"
>
    <div class="local-chatbot-shell">
        <div class="local-chatbot-page">
            <aside class="local-chatbot-sidebar">
            <section class="local-chatbot-card local-chatbot-material-filter">
                <h3><?php echo s(get_string('materialstitle', 'local_chatbot')); ?></h3>
                <p><?php echo s(get_string('uploaddesc', 'local_chatbot')); ?></p>

                <div class="local-chatbot-manual-upload">
                    <h4><?php echo s(get_string('manualuploadtitle', 'local_chatbot')); ?></h4>
                    <label for="local-chatbot-upload-input"><?php echo s(get_string('manualuploadlabel', 'local_chatbot')); ?></label>
                    <input
                        id="local-chatbot-upload-input"
                        class="form-control"
                        type="file"
                        accept=".pdf,.txt,application/pdf,text/plain"
                        multiple
                        <?php echo $canmanualupload ? '' : 'disabled'; ?>
                    />
                    <p class="local-chatbot-upload-help">
                        <?php echo s($canmanualupload
                            ? get_string('manualuploadhelp', 'local_chatbot')
                            : get_string('manualuploadreadonly', 'local_chatbot')); ?>
                    </p>
                    <div class="local-chatbot-upload-actions">
                        <button
                            id="local-chatbot-upload-btn"
                            class="btn btn-primary btn-sm"
                            type="button"
                            <?php echo $canmanualupload ? '' : 'disabled'; ?>
                        >
                            <?php echo s(get_string('manualuploadbutton', 'local_chatbot')); ?>
                        </button>
                        <button
                            id="local-chatbot-clear-upload-btn"
                            class="btn btn-outline-secondary btn-sm"
                            type="button"
                            <?php echo $canmanualupload ? '' : 'disabled'; ?>
                        >
                            <?php echo s(get_string('manualclearbutton', 'local_chatbot')); ?>
                        </button>
                    </div>
                </div>

                <div id="local-chatbot-material-context" class="local-chatbot-material-context local-chatbot-hidden"></div>

                <h4><?php echo s(get_string('topicmaterialstitle', 'local_chatbot')); ?></h4>

                <label for="local-chatbot-chat-class"><?php echo s(get_string('classlabel', 'local_chatbot')); ?></label>
                <select id="local-chatbot-chat-class" class="form-control">
                    <?php if (empty($chatcourses)): ?>
                        <option value=""><?php echo s(get_string('nocoursesavailable', 'local_chatbot')); ?></option>
                    <?php else: ?>
                        <option value=""><?php echo s($classplaceholder); ?></option>
                        <?php foreach ($chatcourses as $course): ?>
                            <?php
                                $courselabel = trim((string)$course->fullname) !== ''
                                    ? $course->fullname
                                    : $course->shortname;
                            ?>
                            <option value="<?php echo (int)$course->id; ?>" data-coursename="<?php echo s($courselabel); ?>">
                                <?php echo s($courselabel); ?>
                            </option>
                        <?php endforeach; ?>
                    <?php endif; ?>
                </select>

                <label for="local-chatbot-chat-topic"><?php echo s(get_string('topiclabel', 'local_chatbot')); ?></label>
                <select id="local-chatbot-chat-topic" class="form-control">
                    <option value=""><?php echo s($topicplaceholder); ?></option>
                </select>
            </section>

            <section class="local-chatbot-card">
                <h3><?php echo s($embeddingconfigtitle); ?></h3>
                <div id="local-chatbot-embedding-config" class="local-chatbot-embedding-config">
                    <div><strong><?php echo s($embeddingconfigactive); ?>:</strong> <?php echo s($activeembeddingtext); ?></div>
                    <div><strong><?php echo s($embeddingconfigbackend); ?>:</strong> <?php echo s($embeddingconfig['default_backend']); ?></div>
                    <div><strong><?php echo s($embeddingconfigollama); ?>:</strong> <?php echo s($embeddingconfig['ollama_model']); ?></div>
                    <div><strong><?php echo s($embeddingconfigbert); ?>:</strong> <?php echo s($embeddingconfig['bert_model']); ?></div>
                </div>
            </section>

            <section class="local-chatbot-card local-chatbot-docs">
                <h3><?php echo s(get_string('uploadedtitle', 'local_chatbot')); ?></h3>
                <div id="local-chatbot-files">
                    <p class="local-chatbot-empty"><?php echo s(get_string('nofiles', 'local_chatbot')); ?></p>
                </div>
            </section>
        </aside>

        <section class="local-chatbot-main">
            <header class="local-chatbot-main-header">
                <h3><?php echo s(get_string('chatheader', 'local_chatbot')); ?></h3>
                <div class="local-chatbot-header-meta">
                    <span id="local-chatbot-usage" class="local-chatbot-usage">
                        <?php echo s(get_string('chatusagelabel', 'local_chatbot')); ?>: 0/80
                    </span>
                    <span id="local-chatbot-status" class="local-chatbot-status">
                        <?php echo s(get_string('statusnodocs', 'local_chatbot')); ?>
                    </span>
                    <button id="local-chatbot-clear" class="local-chatbot-clear btn btn-outline-secondary btn-sm" type="button">
                        <?php echo s(get_string('clearhistorylabel', 'local_chatbot')); ?>
                    </button>
                </div>
            </header>

            <div id="local-chatbot-messages" class="local-chatbot-messages"></div>

            <div class="local-chatbot-composer">
                <input
                    id="local-chatbot-input"
                    type="text"
                    placeholder="<?php echo s(get_string('chatplaceholder', 'local_chatbot')); ?>"
                />
                <button id="local-chatbot-send" class="btn btn-secondary" type="button">
                    <?php echo s(get_string('sendbutton', 'local_chatbot')); ?>
                </button>
            </div>
        </section>

        <section class="local-chatbot-preview">
            <header class="local-chatbot-preview-header">
                <h3><?php echo s(get_string('previewtitle', 'local_chatbot')); ?></h3>
                <div class="local-chatbot-preview-header-meta">
                    <span id="local-chatbot-preview-name" class="local-chatbot-preview-name">-</span>
                    <button id="local-chatbot-refresh-embedding-btn" class="btn btn-outline-secondary btn-sm" type="button">
                        <?php echo s($refreshembeddingbutton); ?>
                    </button>
                </div>
            </header>
            <div id="local-chatbot-preview-embedding-status" class="local-chatbot-preview-embedding-status">
                Select a file to view embedding status.
            </div>
            <div id="local-chatbot-preview-body" class="local-chatbot-preview-body">
                <p class="local-chatbot-empty"><?php echo s(get_string('previewempty', 'local_chatbot')); ?></p>
            </div>
        </section>
    </div>
</div>
</div>
<?php
echo $OUTPUT->footer();

