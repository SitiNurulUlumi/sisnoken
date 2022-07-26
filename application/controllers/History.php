<?php

class History extends CI_Controller
{
    public function __construct()
    {
        parent::__construct();
        $this->load->model('history_model');
        $this->load->helper(array('form', 'url'));
        // load helper
        $this->load->helper(array('form', 'url', 'timezone_helper'));
        $this->load->library(array('session', 'form_validation'));
        if (!$this->session->userdata('is_logged_in')) {
            redirect("home");
        } elseif ($this->session->userdata('role') != 2) {
            redirect("home");
        }
    }

    public function index()
    {
        $data['title'] = 'History Kendaraan Masuk';
        $data['subtitle'] = 'List History Kendaraan Masuk';

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('history/list', $data);
        $this->load->view('templates/admin-footer');
        $this->load->view('history/script', $data);
    }

    public function ajax_list()
    {
        $list = $this->history_model->get_datatables();
        $data = array();
        $no = $_POST['start'];
        foreach ($list as $history) {
            $no++;
            $row = array();
            $row[] = $no;
            $row[] = $history->plat;
            $row[] = tgl_indo($history->masuk);
            $data[] = $row;
        }
        $output = array(
            "draw" => $_POST['draw'],
            "recordsTotal" => $this->history_model->count_all(),
            "recordsFiltered" => $this->history_model->count_filtered(),
            "data" => $data,
        );
        //output to json format
        echo json_encode($output);
    }

    
    public function ajax_keluar($id)
    {
        $data = array(
            'keluar' => timezone(),
        );
        $this->history_model->update(array('id' => $id), $data);
        echo json_encode(array("status" => TRUE));
    }
}
