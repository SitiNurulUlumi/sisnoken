<?php

class Historykeluar extends CI_Controller
{
    public function __construct()
    {
        parent::__construct();
        $this->load->model('historykeluar_model');
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
        $data['title'] = 'History Kendaraan Keluar';
        $data['subtitle'] = 'List History Kendaraan Keluar';

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('history/listkeluar', $data);
        $this->load->view('templates/admin-footer');
        $this->load->view('history/scriptkeluar', $data);
    }

    public function ajax_list()
    {
        $list = $this->historykeluar_model->get_datatables();
        $data = array();
        $no = $_POST['start'];
        foreach ($list as $history) {
            $no++;
            $row = array();
            $row[] = $no;
            $row[] = $history->plat;
            $row[] = tgl_indo($history->keluar);
            $data[] = $row;
        }
        $output = array(
            "draw" => $_POST['draw'],
            "recordsTotal" => $this->historykeluar_model->count_all(),
            "recordsFiltered" => $this->historykeluar_model->count_filtered(),
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
        $this->historykeluar_model->update(array('id' => $id), $data);
        echo json_encode(array("status" => TRUE));
    }
}
