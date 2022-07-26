<?php

class Logs extends CI_Controller
{
    public function __construct()
    {
        parent::__construct();
        $this->load->model('log_model');
        $this->load->helper(array('form', 'url'));
        // load helper
        $this->load->helper(array('form', 'url', 'timezone_helper'));
        $this->load->library(array('session', 'form_validation'));
        if (!$this->session->userdata('is_logged_in')) {
            redirect("home");
        } elseif ($this->session->userdata('role') != 1) {
            redirect("home");
        }
    }

    public function index()
    {
        $data['title'] = 'Log';
        $data['subtitle'] = 'List Log';

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('logs/list', $data);
        $this->load->view('templates/admin-footer');
        $this->load->view('logs/script', $data);
    }

    public function ajax_list()
    {
        $list = $this->log_model->get_datatables();
        $data = array();
        $no = $_POST['start'];
        foreach ($list as $log) {
            if ($log->alert == "danger") {
                $icon = "exclamation";
            } else {
                $icon = "question";
            }
            $no++;
            $row = array();
            $row[] = $no;

            $row[] = '<button class="btn btn-xs btn-' . $log->alert . '" ><i class="fa fa-' . $icon . '"></i></button>';
            $row[] = $log->fullname . '(' . $log->email . ')';
            $row[] = $log->message;
            $row[] = tgl_indo($log->created_at);
            $data[] = $row;
        }
        $output = array(
            "draw" => $_POST['draw'],
            "recordsTotal" => $this->log_model->count_all(),
            "recordsFiltered" => $this->log_model->count_filtered(),
            "data" => $data,
        );
        //output to json format
        echo json_encode($output);
    }
}
