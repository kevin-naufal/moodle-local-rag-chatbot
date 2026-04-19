<?php
defined('MOODLE_INTERNAL') || die();

/**
 * Adds LLM Chat page to navigation.
 *
 * @param global_navigation $navigation
 * @return void
 */
function local_chatbot_extend_navigation(global_navigation $navigation): void {
    global $USER;

    if (!isloggedin() || isguestuser()) {
        return;
    }

    $context = context_system::instance();
    if (!has_capability('local/chatbot:view', $context)) {
        return;
    }

    $url = new moodle_url('/local/chatbot/index.php');
    $node = navigation_node::create(
        get_string('pluginname', 'local_chatbot'),
        $url,
        navigation_node::TYPE_CUSTOM,
        null,
        'local_chatbot'
    );
    $navigation->add_node($node);

    require_once(__DIR__ . '/locallib.php');
    $isteacher = local_chatbot_user_is_teacher_like((int)$USER->id) || is_siteadmin((int)$USER->id);
    if ($isteacher) {
        $reporturl = new moodle_url('/local/chatbot/teacher_report.php');
        $reportnode = navigation_node::create(
            get_string('teacherreportlink', 'local_chatbot'),
            $reporturl,
            navigation_node::TYPE_CUSTOM,
            null,
            'local_chatbot_teacher_report'
        );
        $node->add_node($reportnode);

        $weighturl = new moodle_url('/local/chatbot/weights.php');
        $weightnode = navigation_node::create(
            get_string('weightsettingslink', 'local_chatbot'),
            $weighturl,
            navigation_node::TYPE_CUSTOM,
            null,
            'local_chatbot_weight_settings'
        );
        $node->add_node($weightnode);
    }
}

/**
 * Adds weight-settings shortcut into course navigation for teacher/admin.
 *
 * @param navigation_node $navigation
 * @param stdClass $course
 * @param context_course $context
 * @return void
 */
function local_chatbot_extend_navigation_course(
    navigation_node $navigation,
    stdClass $course,
    context_course $context
): void {
    global $USER;

    if (!isloggedin() || isguestuser() || empty($course->id) || (int)$course->id === SITEID) {
        return;
    }

    $systemcontext = context_system::instance();
    if (!has_capability('local/chatbot:view', $systemcontext, (int)$USER->id)) {
        return;
    }

    require_once(__DIR__ . '/locallib.php');
    $canmanage = has_capability('local/chatbot:manageweights', $context, (int)$USER->id) ||
        has_capability('moodle/course:update', $context, (int)$USER->id) ||
        local_chatbot_user_is_teacher_like((int)$USER->id) ||
        is_siteadmin((int)$USER->id);
    if (!$canmanage) {
        return;
    }

    $weighturl = new moodle_url('/local/chatbot/weights.php', ['courseid' => (int)$course->id]);
    $navigation->add(
        get_string('weightsettingslink', 'local_chatbot'),
        $weighturl,
        navigation_node::TYPE_SETTING,
        null,
        'local_chatbot_course_weight_settings'
    );
}

/**
 * Add native activity-form fields for assessment type and weight label.
 *
 * @param moodleform_mod $formwrapper
 * @param MoodleQuickForm $mform
 * @return void
 */
function local_chatbot_coursemodule_standard_elements($formwrapper, $mform): void {
    if (!class_exists('\local_chatbot\service\weight_ui_service')) {
        return;
    }
    if (!has_capability('moodle/course:manageactivities', $formwrapper->get_context())) {
        return;
    }

    $modulename = local_chatbot_native_weighting_modulename($formwrapper);
    if (!in_array($modulename, ['assign', 'quiz'], true)) {
        return;
    }

    $typeoptions = local_chatbot_native_weighting_type_options($modulename);
    if (empty($typeoptions)) {
        return;
    }

    $mform->addElement('header', 'local_chatbot_weighting_header', get_string('nativeweightingheading', 'local_chatbot'));

    $mform->addElement(
        'select',
        'local_chatbot_activity_type',
        get_string('nativeweightingactivitytype', 'local_chatbot'),
        $typeoptions
    );
    $mform->setType('local_chatbot_activity_type', PARAM_ALPHANUMEXT);
    $mform->addHelpButton('local_chatbot_activity_type', 'nativeweightingactivitytype', 'local_chatbot');

    $mform->addElement(
        'select',
        'local_chatbot_weight_label',
        get_string('assignmentweightlabel', 'local_chatbot'),
        local_chatbot_native_weight_label_options()
    );
    $mform->setType('local_chatbot_weight_label', PARAM_TEXT);
    $mform->addHelpButton('local_chatbot_weight_label', 'assignmentweightlabel', 'local_chatbot');

    $defaulttype = local_chatbot_native_weighting_default_type($modulename);
    $defaultlabel = \local_chatbot\service\weight_ui_service::WEIGHT_LABEL_MEDIUM;
    $cm = $formwrapper->get_coursemodule();
    if ($cm && !empty($cm->id) && \local_chatbot\service\weight_ui_service::tables_ready()) {
        try {
            $course = $formwrapper->get_course();
            $scheme = \local_chatbot\service\weight_ui_service::get_or_create_active_scheme((int)$course->id);
            $maps = \local_chatbot\service\weight_ui_service::get_activity_maps((int)$scheme->id);
            if (isset($maps[(int)$cm->id])) {
                $mapped = $maps[(int)$cm->id];
                $mappedtype = trim((string)($mapped->subtype ?? ''));
                if (array_key_exists($mappedtype, $typeoptions)) {
                    $defaulttype = $mappedtype;
                }
                $mappedweight = (float)($mapped->item_weight_percent ?? 100.0);
                $defaultlabel = \local_chatbot\service\weight_ui_service::weight_label_from_percent($mappedweight);
            }
        } catch (\Throwable $e) {
            debugging('local_chatbot native weighting form default failed: ' . $e->getMessage(), DEBUG_DEVELOPER);
        }
    }

    $mform->setDefault('local_chatbot_activity_type', $defaulttype);
    $mform->setDefault('local_chatbot_weight_label', $defaultlabel);
}

/**
 * Persist native activity-form weighting selections after save.
 *
 * @param stdClass $moduleinfo
 * @param stdClass $course
 * @return stdClass
 */
function local_chatbot_coursemodule_edit_post_actions($moduleinfo, $course) {
    if (!class_exists('\local_chatbot\service\weight_ui_service')) {
        return $moduleinfo;
    }

    $modulename = trim((string)($moduleinfo->modulename ?? ''));
    if (!in_array($modulename, ['assign', 'quiz'], true)) {
        return $moduleinfo;
    }
    if (empty($moduleinfo->coursemodule) || empty($course->id)) {
        return $moduleinfo;
    }
    if (!isset($moduleinfo->local_chatbot_activity_type) || !isset($moduleinfo->local_chatbot_weight_label)) {
        return $moduleinfo;
    }

    $activitytype = local_chatbot_native_weighting_normalize_type(
        $modulename,
        (string)$moduleinfo->local_chatbot_activity_type
    );
    $weightlabel = \local_chatbot\service\weight_ui_service::normalize_weight_label(
        (string)$moduleinfo->local_chatbot_weight_label
    );

    $payload = [
        'weight_bucket_type' => $activitytype,
        'weight_source' => \local_chatbot\service\weight_ui_service::SOURCE_TEACHER,
        'activity_weight_label' => $weightlabel,
        'activity_weight_percent' => \local_chatbot\service\weight_ui_service::weight_percent_from_label($weightlabel),
    ];

    try {
        \local_chatbot\service\weight_ui_service::apply_map_from_draft_payload(
            (int)$course->id,
            (int)$moduleinfo->coursemodule,
            $modulename,
            trim((string)($moduleinfo->name ?? '')),
            $payload
        );
    } catch (\Throwable $e) {
        debugging('local_chatbot native weighting save failed: ' . $e->getMessage(), DEBUG_DEVELOPER);
    }

    return $moduleinfo;
}

/**
 * Resolve current native form module name.
 *
 * @param moodleform_mod $formwrapper
 * @return string
 */
function local_chatbot_native_weighting_modulename($formwrapper): string {
    $current = $formwrapper->get_current();
    if (is_object($current) && !empty($current->modulename)) {
        return trim((string)$current->modulename);
    }
    $cm = $formwrapper->get_coursemodule();
    if ($cm && !empty($cm->modname)) {
        return trim((string)$cm->modname);
    }
    return '';
}

/**
 * Type option list for native module forms.
 *
 * @param string $modulename
 * @return array<string,string>
 */
function local_chatbot_native_weighting_type_options(string $modulename): array {
    if ($modulename === 'assign') {
        return [
            \local_chatbot\service\weight_ui_service::TYPE_INDIVIDUAL => get_string('weighttypeindividual', 'local_chatbot'),
            \local_chatbot\service\weight_ui_service::TYPE_GROUP => get_string('weighttypegroup', 'local_chatbot'),
        ];
    }
    if ($modulename === 'quiz') {
        return [
            \local_chatbot\service\weight_ui_service::TYPE_PRACTICE => get_string('weighttypepractice', 'local_chatbot'),
            \local_chatbot\service\weight_ui_service::TYPE_QUIZ => get_string('weighttypequiz', 'local_chatbot'),
            \local_chatbot\service\weight_ui_service::TYPE_UTS => get_string('weighttypeuts', 'local_chatbot'),
            \local_chatbot\service\weight_ui_service::TYPE_UAS => get_string('weighttypeuas', 'local_chatbot'),
        ];
    }
    return [];
}

/**
 * Default assessment type for one native module.
 *
 * @param string $modulename
 * @return string
 */
function local_chatbot_native_weighting_default_type(string $modulename): string {
    if ($modulename === 'assign') {
        return \local_chatbot\service\weight_ui_service::TYPE_INDIVIDUAL;
    }
    if ($modulename === 'quiz') {
        return \local_chatbot\service\weight_ui_service::TYPE_QUIZ;
    }
    return \local_chatbot\service\weight_ui_service::TYPE_QUIZ;
}

/**
 * Normalize submitted type value by module.
 *
 * @param string $modulename
 * @param string $raw
 * @return string
 */
function local_chatbot_native_weighting_normalize_type(string $modulename, string $raw): string {
    $options = local_chatbot_native_weighting_type_options($modulename);
    $candidate = trim((string)$raw);
    if (array_key_exists($candidate, $options)) {
        return $candidate;
    }
    return local_chatbot_native_weighting_default_type($modulename);
}

/**
 * Weight label option list with percent suffix.
 *
 * @return array<string,string>
 */
function local_chatbot_native_weight_label_options(): array {
    $map = \local_chatbot\service\weight_ui_service::weight_label_map();
    return [
        \local_chatbot\service\weight_ui_service::WEIGHT_LABEL_VERY_EASY =>
            get_string('assignmentweightveryeasy', 'local_chatbot') . ' (' . format_float((float)$map[\local_chatbot\service\weight_ui_service::WEIGHT_LABEL_VERY_EASY], 0) . '%)',
        \local_chatbot\service\weight_ui_service::WEIGHT_LABEL_EASY =>
            get_string('assignmentweighteasy', 'local_chatbot') . ' (' . format_float((float)$map[\local_chatbot\service\weight_ui_service::WEIGHT_LABEL_EASY], 0) . '%)',
        \local_chatbot\service\weight_ui_service::WEIGHT_LABEL_MEDIUM =>
            get_string('assignmentweightmedium', 'local_chatbot') . ' (' . format_float((float)$map[\local_chatbot\service\weight_ui_service::WEIGHT_LABEL_MEDIUM], 0) . '%)',
        \local_chatbot\service\weight_ui_service::WEIGHT_LABEL_HARD =>
            get_string('assignmentweighthard', 'local_chatbot') . ' (' . format_float((float)$map[\local_chatbot\service\weight_ui_service::WEIGHT_LABEL_HARD], 0) . '%)',
        \local_chatbot\service\weight_ui_service::WEIGHT_LABEL_VERY_HARD =>
            get_string('assignmentweightveryhard', 'local_chatbot') . ' (' . format_float((float)$map[\local_chatbot\service\weight_ui_service::WEIGHT_LABEL_VERY_HARD], 0) . '%)',
    ];
}

/**
 * Add mastery summary nodes to user profile page.
 *
 * @param \core_user\output\myprofile\tree $tree
 * @param stdClass $user
 * @param bool $iscurrentuser
 * @param stdClass|null $course
 * @return void
 */
function local_chatbot_myprofile_navigation(\core_user\output\myprofile\tree $tree, $user, $iscurrentuser, $course): void {
    global $CFG, $USER;

    if (!isloggedin() || isguestuser()) {
        return;
    }

    $systemcontext = context_system::instance();
    if (!has_capability('local/chatbot:view', $systemcontext)) {
        return;
    }

    require_once($CFG->dirroot . '/local/chatbot/locallib.php');
    if (!local_chatbot_learning_tables_ready()) {
        return;
    }

    $topicrows = local_chatbot_get_student_mastery_rows((int)$user->id);
    $classrows = local_chatbot_get_student_class_mastery_rows((int)$user->id);
    $overall = local_chatbot_get_student_overall_mastery((int)$user->id);
    $content = local_chatbot_render_profile_mastery_html(
        $topicrows,
        $classrows,
        $overall,
        $iscurrentuser && (local_chatbot_user_is_teacher_like((int)$USER->id) || is_siteadmin()),
        (int)$user->id,
        (local_chatbot_user_is_teacher_like((int)$USER->id) || is_siteadmin((int)$USER->id))
    );
    if (trim($content) === '') {
        return;
    }

    $masterynode = new \core_user\output\myprofile\node(
        'reports',
        'chatbotmastery',
        get_string('dashboardtitle', 'local_chatbot'),
        null,
        null,
        $content
    );
    $tree->add_node($masterynode);

    $chaturl = new moodle_url('/local/chatbot/index.php');
    $chatnode = new \core_user\output\myprofile\node(
        'reports',
        'chatbotopen',
        get_string('pluginname', 'local_chatbot'),
        null,
        $chaturl
    );
    $tree->add_node($chatnode);
}

/**
 * Render profile mastery HTML snippet.
 *
 * @param array $topicrows
 * @param array $classrows
 * @param array $overall
 * @param bool $showteacheroverview
 * @return string
 */
function local_chatbot_render_profile_mastery_html(
    array $topicrows,
    array $classrows,
    array $overall,
    bool $showteacheroverview = false,
    int $userid = 0,
    bool $vieweristeacher = false
): string {
    global $USER;

    if (empty($topicrows) && empty($classrows) && !$showteacheroverview) {
        return html_writer::tag('p', get_string('dashboardempty', 'local_chatbot'));
    }

    $html = '';
    if (!empty($topicrows) || !empty($classrows)) {
        $html .= html_writer::tag('h4', get_string('dashboardsectionoverall', 'local_chatbot'));
        $overallmetrics = [
            get_string('dashboardcardavgmastery', 'local_chatbot') . ': ' .
                format_float((float)($overall['overallmastery'] ?? 0), 1) . '%',
            get_string('dashboardcardoverallaccuracy', 'local_chatbot') . ': ' .
                format_float((float)($overall['overallaccuracy'] ?? 0), 1) . '%',
            get_string('dashboardcardtrackedclasses', 'local_chatbot') . ': ' . (int)($overall['classcount'] ?? 0),
            get_string('dashboardcardtrackedtopics', 'local_chatbot') . ': ' . (int)($overall['topiccount'] ?? 0),
            get_string('dashboardcardattempts', 'local_chatbot') . ': ' . (int)($overall['attemptsum'] ?? 0),
            get_string('dashboardcardlastupdate', 'local_chatbot') . ': ' .
                ((int)($overall['lastupdate'] ?? 0) > 0
                    ? userdate((int)$overall['lastupdate'], get_string('strftimedatetime', 'langconfig'))
                    : '-'),
        ];
        $html .= html_writer::alist($overallmetrics);

        $html .= html_writer::tag('h4', get_string('dashboardsectionclasses', 'local_chatbot'), ['class' => 'mt-3']);
        $classtable = new html_table();
        $classtable->head = [
            get_string('dashboardtablecourse', 'local_chatbot'),
            get_string('dashboardtablemastery', 'local_chatbot'),
            get_string('dashboardtableaccuracy', 'local_chatbot'),
            get_string('dashboardtabletopics', 'local_chatbot'),
            get_string('dashboardtableattempts', 'local_chatbot'),
            get_string('dashboardtableupdated', 'local_chatbot'),
        ];
        $classtable->data = [];
        foreach ($classrows as $row) {
            $courselabel = trim((string)$row->fullname) !== '' ? (string)$row->fullname : (string)$row->shortname;
            $classtable->data[] = [
                s($courselabel),
                s(format_float((float)$row->classmastery, 1) . '%'),
                s(format_float((float)$row->classaccuracy, 1) . '%'),
                (int)$row->topiccount,
                (int)$row->attemptsum,
                s((int)$row->lastupdate > 0
                    ? userdate((int)$row->lastupdate, get_string('strftimedatetime', 'langconfig'))
                    : '-'),
            ];
        }
        if (empty($classtable->data)) {
            $classtable->data[] = [get_string('dashboardempty', 'local_chatbot'), '', '', '', '', ''];
        }
        $html .= html_writer::table($classtable);

        $html .= html_writer::tag(
            'h4',
            get_string('dashboardsectionstudenttopics', 'local_chatbot'),
            ['class' => 'mt-3']
        );
        $canseeprogressinsights = ($userid > 0 && (int)$USER->id === $userid) || $vieweristeacher;
        $progressrows = $canseeprogressinsights ? local_chatbot_get_student_topic_progress_rows($userid, 75.0) : [];
        $progressmap = [];
        foreach ($progressrows as $prow) {
            $progresskey = (int)$prow->courseid . '|' . (string)$prow->topic;
            $progressmap[$progresskey] = $prow;
        }

        $table = new html_table();
        $table->head = [
            get_string('dashboardtablecourse', 'local_chatbot'),
            get_string('dashboardtabletopic', 'local_chatbot'),
            get_string('dashboardtablemastery', 'local_chatbot'),
            get_string('dashboardtableaccuracy', 'local_chatbot'),
            get_string('dashboardtableattempts', 'local_chatbot'),
            get_string('dashboardtablemasterydelta', 'local_chatbot'),
            get_string('dashboardtablefirstattempt', 'local_chatbot'),
            get_string('dashboardtabletrendchart', 'local_chatbot'),
            get_string('dashboardtableupdated', 'local_chatbot'),
        ];
        $table->data = [];

        foreach (array_slice($topicrows, 0, 20) as $row) {
            $courselabel = trim((string)$row->fullname) !== '' ? (string)$row->fullname : (string)$row->shortname;
            $rowkey = (int)$row->courseid . '|' . (string)$row->topic;
            $progress = $progressmap[$rowkey] ?? null;
            $delta = $progress ? (float)$progress->mastery_change : 0.0;
            $deltasign = $delta >= 0 ? '+' : '';
            $firstattempt = ($progress && $progress->first_attempt_accuracy !== null)
                ? format_float((float)$progress->first_attempt_accuracy, 1) . '%'
                : '-';
            $trendchart = $progress
                ? local_chatbot_render_snapshot_trend_chart((array)$progress->trend_points)
                : '-';
            $table->data[] = [
                s($courselabel),
                s((string)$row->topic),
                s(format_float((float)$row->mastery, 1) . '%'),
                s(format_float((float)$row->accuracy_avg, 1) . '%'),
                (int)$row->attempt_count,
                $progress ? s($deltasign . format_float($delta, 1) . 'pp') : '-',
                s($firstattempt),
                $trendchart,
                s((int)$row->timemodified > 0
                    ? userdate((int)$row->timemodified, get_string('strftimedatetime', 'langconfig'))
                    : '-'),
            ];
        }
        $html .= html_writer::table($table);
    } else {
        $html .= html_writer::tag('p', get_string('dashboardempty', 'local_chatbot'));
    }

    if ($showteacheroverview) {
        require_once(__DIR__ . '/../../enrol/locallib.php');
        $courseids = [];
        $courses = enrol_get_users_courses((int)$USER->id, true, 'id', 'sortorder ASC');
        foreach ($courses as $course) {
            $coursecontext = context_course::instance((int)$course->id, IGNORE_MISSING);
            if ($coursecontext && has_capability('moodle/course:update', $coursecontext, (int)$USER->id)) {
                $courseids[] = (int)$course->id;
            }
        }

        if (empty($courseids) && is_siteadmin((int)$USER->id)) {
            global $DB;
            $courseids = array_map(
                'intval',
                array_keys($DB->get_records_menu('local_chatbot_std_profile', null, '', 'courseid,courseid'))
            );
        }

        if (!empty($courseids)) {
            $overview = local_chatbot_get_teacher_mastery_dashboard($courseids);
            if ((int)$overview['summary']['profilecount'] > 0) {
                $learners = array_values((array)($overview['learners'] ?? []));

                $teacheritems = [
                    get_string('dashboardcardstudents', 'local_chatbot') . ': ' . (int)$overview['summary']['studentcount'],
                    get_string('dashboardcardprofiles', 'local_chatbot') . ': ' . (int)$overview['summary']['profilecount'],
                    get_string('dashboardcardavgmastery', 'local_chatbot') . ': ' .
                        format_float((float)$overview['summary']['avgmastery'], 1) . '%',
                    get_string('dashboardcardevents', 'local_chatbot') . ': ' . (int)$overview['summary']['eventcount'],
                ];
                $html .= html_writer::tag(
                    'h4',
                    get_string('dashboardsubtitleteacher', 'local_chatbot'),
                    ['class' => 'mt-3']
                );
                $html .= html_writer::alist($teacheritems);

                $html .= html_writer::tag(
                    'h4',
                    get_string('dashboardsectionteacherstudents', 'local_chatbot'),
                    ['class' => 'mt-3']
                );

                $studenttable = new html_table();
                $studenttable->head = [
                    get_string('dashboardtablecourse', 'local_chatbot'),
                    get_string('dashboardtablestudent', 'local_chatbot'),
                    get_string('dashboardtablemastery', 'local_chatbot'),
                    get_string('dashboardtableaccuracy', 'local_chatbot'),
                    get_string('dashboardtableattempts', 'local_chatbot'),
                    get_string('dashboardtableupdated', 'local_chatbot'),
                ];
                $studenttable->data = [];

                foreach (array_slice($learners, 0, 100) as $row) {
                    $courselabel = trim((string)$row->fullname) !== '' ? (string)$row->fullname : (string)$row->shortname;
                    $studentname = trim((string)$row->firstname . ' ' . (string)$row->lastname);
                    if ($studentname === '') {
                        $studentname = get_string('unknownuser', 'moodle');
                    }
                    $profileurl = new moodle_url('/user/profile.php', ['id' => (int)$row->userid]);

                    $studenttable->data[] = [
                        s($courselabel),
                        html_writer::link($profileurl, s($studentname)),
                        s(format_float((float)$row->avgmastery, 1) . '%'),
                        s(format_float((float)$row->avgaccuracy, 1) . '%'),
                        (int)$row->attemptsum,
                        s((int)$row->lastupdate > 0
                            ? userdate((int)$row->lastupdate, get_string('strftimedatetime', 'langconfig'))
                            : '-'),
                    ];
                }

                if (empty($studenttable->data)) {
                    $studenttable->data[] = [get_string('dashboardempty', 'local_chatbot'), '', '', '', '', ''];
                }

                $html .= html_writer::table($studenttable);

                $reporturl = new moodle_url('/local/chatbot/teacher_report.php');
                $html .= html_writer::tag(
                    'p',
                    html_writer::link($reporturl, get_string('teacherreportlink', 'local_chatbot'))
                );
            }
        }
    }

    return $html;
}
