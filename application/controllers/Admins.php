<?php

class Admins extends CI_Controller
{

    public function __construct()
    {
        parent::__construct();
        $this->load->model('user_model');
        $this->load->helper(array('form', 'url', 'timezone_helper'));
        $this->load->library(array('session', 'form_validation'));
        $this->load->library('form_validation');
        if (!$this->session->userdata('is_logged_in')) {
            redirect("home");
        } elseif ($this->session->userdata('role') == 3) {
            redirect("auth/logout");
        }
    }

    public function index()
    {
        $this->list();
    }

    public function list()
    {
        $data['title'] = 'Data Admin';
        $data['subtitle'] = 'List Admin';
        $data['role_list'] = '2';

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('users/list', $data);
        $this->load->view('templates/admin-footer');
        $this->load->view('users/script', $data);
    }
}
