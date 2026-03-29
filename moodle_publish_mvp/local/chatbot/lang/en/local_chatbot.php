<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Moodle is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Moodle.  If not, see <http://www.gnu.org/licenses/>.

defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'LLM Chat';
$string['settingsheading'] = 'LLM Chat settings';
$string['projectpath'] = 'Project path';
$string['projectpath_desc'] = 'Absolute path to the LLM project folder.';
$string['pythonpath'] = 'Python executable path';
$string['pythonpath_desc'] = 'Absolute path to python executable used by the plugin.';
$string['runnerfile'] = 'Runner file name';
$string['runnerfile_desc'] = 'Backend runner filename inside the project path.';

$string['chatbot:view'] = 'View LLM Chat page.';
$string['chatbot:managedrafts'] = 'Create and update generated drafts.';
$string['chatbot:publish'] = 'Publish generated draft to course.';

$string['uploadtitle'] = 'Insert PDF/TXT';
$string['uploaddesc'] = 'Select class and topic to load materials from course resources.';
$string['uploadbutton'] = 'Upload selected files';
$string['uploadedtitle'] = 'Materials';
$string['nofiles'] = 'No materials found for selected class/topic.';
$string['statusready'] = 'RAG ready';
$string['statusnodocs'] = 'No materials selected';
$string['chatheader'] = 'Chat with class materials';
$string['chatplaceholder'] = 'Ask a question about selected class materials...';
$string['chatusagelabel'] = 'Usage';
$string['clearhistorylabel'] = 'Clear history';
$string['clearhistoryconfirm'] = 'Clear this chat history?';
$string['defaultgreeting'] = 'Hello. Select class and topic in the left panel, then ask about the materials.';
$string['thinking'] = 'Thinking...';
$string['uploading'] = 'Uploading...';
$string['uploadfailed'] = 'Upload failed.';
$string['chaterror'] = 'Failed to process chat request.';
$string['previewtitle'] = 'Preview Draft';
$string['previewempty'] = 'No preview available.';
$string['previewloading'] = 'Generating preview...';
$string['previewerror'] = 'Failed to generate preview.';

$string['tabchat'] = 'Chat';
$string['tabassignment'] = 'Assignment';
$string['tabpractice'] = 'Practice';
$string['roleteacheronly'] = 'This feature is available for teachers only.';

$string['assignmenttitle'] = 'Create assignment with LLM';
$string['assignmentdesc'] = 'Generate assignment draft, review it, then publish to target class.';
$string['assignmentclass'] = 'Target class';
$string['assignmentclassplaceholder'] = 'Select class';
$string['assignmentnocourses'] = 'No teacher course found';
$string['assignmenttopic'] = 'Topic';
$string['assignmenttopicplaceholder'] = 'Select topic';
$string['assignmenttopicloading'] = 'Loading topics...';
$string['assignmenttopicempty'] = 'No topics found in this class';
$string['assignmentpdf'] = 'Material';
$string['assignmentpdfplaceholder'] = 'Select PDF';
$string['assignmentpdfloading'] = 'Loading PDFs...';
$string['assignmentpdfempty'] = 'No PDF resource found in this class';
$string['assignmenttype'] = 'Assignment type';
$string['assignmentcount'] = 'Number of questions/components';
$string['assignmentnotes'] = 'Additional notes';
$string['assignmentgenerate'] = 'Generate';
$string['assignmentregenerate'] = 'Regenerate';
$string['assignmentpublish'] = 'Publish';
$string['assignmentpublished'] = 'Published';
$string['assignmentpublishing'] = 'Publishing...';
$string['assignmentpublisherror'] = 'Failed to publish draft.';
$string['assignmentgeneratedfirst'] = 'Generate draft first before publishing.';
$string['assignmentselectclassfirst'] = 'Select target class first.';
$string['assignmentpreview'] = 'Assignment draft preview';
$string['assignmentplaceholder'] = 'Draft will appear here after generation.';

$string['practicetitle'] = 'Practice generator';
$string['practicedesc'] = 'Generate practice questions from selected topic.';
$string['practicegenerate'] = 'Generate practice';
$string['practicepreview'] = 'Practice preview';
$string['practiceplaceholder'] = 'Practice output will appear here.';

$string['savedraftsuccess'] = 'Draft saved successfully.';
$string['publishsuccess'] = 'Draft published successfully to class.';
$string['draftcoursemismatch'] = 'Draft does not belong to the selected course.';
$string['publishfailed'] = 'Publish failed while creating Moodle activity.';
$string['missingdraftpayload'] = 'No draft payload was provided.';
$string['invaliddraftjson'] = 'Draft JSON is invalid.';
$string['missingdraftsection'] = 'Draft is missing required section: {$a}.';
$string['invalidassignmenttitle'] = 'Assignment title is empty or invalid.';
$string['invalidquestions'] = 'Question list is empty or invalid.';
$string['questioncountmismatch'] = 'Question count does not match configured value.';
$string['invalidquestionstem'] = 'Question stem is invalid at number {$a}.';
$string['invalidquestionoptions'] = 'Question options A-D are invalid at number {$a}.';
$string['invalidanswerkey'] = 'Answer key is invalid.';
$string['invalidanswerkeynumber'] = 'Answer key is missing question number {$a}.';
$string['invalidanswerkeyletter'] = 'Answer key letter is invalid at question {$a}.';
