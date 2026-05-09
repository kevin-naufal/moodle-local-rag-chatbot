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

$classplaceholder = get_string('classplaceholder', 'local_chatbot');
$topicplaceholder = get_string('topicplaceholder', 'local_chatbot');
if (preg_match('/^\[\[[^\]]+\]\]$/', $classplaceholder)) {
    $classplaceholder = 'Select class';
}
if (preg_match('/^\[\[[^\]]+\]\]$/', $topicplaceholder)) {
    $topicplaceholder = 'Select topic';
}

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

$PAGE->requires->css('/local/chatbot/styles.css');
$PAGE->requires->js_call_amd('local_chatbot/widget', 'init', [[
    'ajaxurl' => (new moodle_url('/local/chatbot/ajax.php'))->out(false),
    'sesskey' => sesskey(),
    'userid' => (int)$USER->id,
    'chaterror' => get_string('chaterror', 'local_chatbot'),
    'nofiles' => get_string('nofiles', 'local_chatbot'),
    'defaultgreeting' => get_string('defaultgreeting', 'local_chatbot'),
    'thinking' => get_string('thinking', 'local_chatbot'),
    'chatusagelabel' => get_string('chatusagelabel', 'local_chatbot'),
    'previewempty' => get_string('previewempty', 'local_chatbot'),
    'previewloading' => get_string('previewloading', 'local_chatbot'),
    'previewerror' => get_string('previewerror', 'local_chatbot'),
    'previewopenpdf' => get_string('previewopenpdf', 'local_chatbot'),
    'previewpdffallback' => get_string('previewpdffallback', 'local_chatbot'),
    'clearhistoryconfirm' => get_string('clearhistoryconfirm', 'local_chatbot'),
    'statusready' => get_string('statusready', 'local_chatbot'),
    'statusnodocs' => get_string('statusnodocs', 'local_chatbot'),
    'courseclassplaceholder' => $classplaceholder,
    'coursetopicplaceholder' => $topicplaceholder,
    'coursetopics' => $coursetopicsmap,
]]);

echo $OUTPUT->header();
?>
<div class="local-chatbot-shell">
    <div class="local-chatbot-page">
        <aside class="local-chatbot-sidebar">
            <section class="local-chatbot-card local-chatbot-material-filter">
                <h3><?php echo s(get_string('materialstitle', 'local_chatbot')); ?></h3>
                <p><?php echo s(get_string('uploaddesc', 'local_chatbot')); ?></p>

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
                <span id="local-chatbot-preview-name" class="local-chatbot-preview-name">-</span>
            </header>
            <div id="local-chatbot-preview-body" class="local-chatbot-preview-body">
                <p class="local-chatbot-empty"><?php echo s(get_string('previewempty', 'local_chatbot')); ?></p>
            </div>
        </section>
    </div>
</div>
<?php
echo $OUTPUT->footer();

